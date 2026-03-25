import subprocess
import os
import sys
import sounddevice as sd
import soundfile as sf
import time
import pyaudio
import numpy as np
from faster_whisper import WhisperModel
import requests


# --- CONFIGURATION ---
print("--- UR3e Voice System Initialization ---")

# --- LM STUDIO CONFIGURATION ---
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

# TTS Config
PIPER_EXE = "piper/piper.exe"  # Adjust to "./piper/piper.exe" if you kept the folder structure
MODEL_PATH = "models/en_US-amy-medium.onnx"
TEMP_WAV = "last_speech.wav"

# STT Config
MODEL_SIZE = "tiny.en"
DEVICE = "cpu" 

print("Loading Ear Engine (Whisper)...")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type="int8")

# --- FUNCTIONS ---
def verify_microphone():
    """Returns True if a valid input device is found, else False."""
    p = pyaudio.PyAudio()
    try:
        # 1. Check for any input devices at all
        count = p.get_device_count()
        if count == 0:
            print("❌ CRITICAL: No audio devices found on this PC.")
            return False

        # 2. Get the default input device info
        try:
            default_mic = p.get_default_input_device_info()
            print(f"Mic Detected: {default_mic['name']}")
            
            # 3. Check if it actually supports input channels
            if default_mic['maxInputChannels'] > 0:
                return True
            else:
                print("ERROR: Default device found, but it has 0 input channels.")
                return False
                
        except OSError:
            print("WINDOWS ERROR: No Default Input Device is set.")
            print("   -> Go to Sound Settings and set your microphone as 'Default'.")
            return False
            
    finally:
        p.terminate()

def robot_speak(text, speed=0.8):
    if not os.path.exists(PIPER_EXE):
        print(f"ERROR: {PIPER_EXE} missing!")
        return

    try:
        command = [PIPER_EXE, "--model", MODEL_PATH, "--length_scale", str(speed), "--output_file", TEMP_WAV]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        process.communicate(input=text)

        if os.path.exists(TEMP_WAV):
            data, fs = sf.read(TEMP_WAV, dtype='float32')
            sd.play(data, fs)
            sd.wait()
            os.remove(TEMP_WAV)
    except Exception as e:
        print(f"Speech Error: {e}")

def listen_smart():
    chunk = 1024
    fs = 16000
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True, frames_per_buffer=chunk)
        print("\n👂 Listening... (Stop speaking to process)")
        
        frames = []
        silent_chunks = 0
        # 1.2 seconds of silence = stop (approx 18 chunks)
        SILENCE_THRESHOLD = 18 
        # Safety limit: 15 seconds max recording
        MAX_CHUNKS = int(fs / chunk * 15)

        while len(frames) < MAX_CHUNKS:
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)
            
            # Volume check for auto-stop
            audio_chunk = np.frombuffer(data, dtype=np.int16)
            if np.max(np.abs(audio_chunk)) < 500: # Adjust 500 if your room is noisy
                silent_chunks += 1
            else:
                silent_chunks = 0
            
            # If we've recorded at least 1 second and now it's silent, stop
            if len(frames) > 15 and silent_chunks > SILENCE_THRESHOLD:
                break

        stream.stop_stream()
        stream.close()
        
        # --- THE MISSING TRANSCRIPTION LOGIC ---
        if not frames:
            return ""

        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0
        
        segments, _ = model.transcribe(
            audio_data, 
            beam_size=1,        # Faster processing
            vad_filter=True,    # Cleaner results
            language="en"
        )
        
        return "".join([s.text for s in segments]).strip()

    except Exception as e:
        print(f"STT Error: {e}")
        return ""
    finally:
        p.terminate()

def query_rag(prompt):
    """Communicates with the local LM Studio / RAG Server."""
    payload = {
        "model": "local-model", # LM Studio usually ignores this and uses the loaded model
        "messages": [
            {"role": "system", "content": "You are a robotic assistant. Give concise answers."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
    }
    try:
        # 30s timeout is safer for local LLMs which can be slow on CPU
        response = requests.post(LM_STUDIO_URL, json=payload, timeout=30)
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Communication error with brain engine: {e}"
    

# --- MAIN LOOP ---
if __name__ == "__main__":
    # 1. Hardware Check
    if not verify_microphone():
        robot_speak("Stopping script due to microphone issues.")
        exit()

   # 2. Initialization
    robot_speak("System online and listening.")
    
    # 3. The Main Loop
    while True:
        command = listen_smart()
        
        if command:
            print(f"User Input: {command}")
            cmd_clean = command.lower()

            # --- 1. TERMINATION COMMAND ---
            if any(x in cmd_clean for x in ["terminate program", "stop all systems"]):
                robot_speak("Terminating all systems. Goodbye.")
                time.sleep(0.5) 
                sys.exit()

            # --- 2. RAG QUERY LOGIC ---
            # Trigger: "Query"
            elif "query" in cmd_clean:
                # Remove the trigger word to get the clean question
                clean_query = cmd_clean.replace("query", "").strip()
                
                print(f"🧠 Querying RAG: {clean_query}")
                result = query_rag(clean_query)

                # Check preference: "Output Voice" vs default text
                if "output voice" in cmd_clean or "voice" in cmd_clean:
                    clean_query = cmd_clean.replace("output voice", "").strip()
                    clean_query = cmd_clean.replace("voice", "").strip()
                    robot_speak(result)
                else:
                    print(f"\n[LLM RESPONSE]\n{result}\n")
                    robot_speak("Displaying result on screen.")

            # --- 3. ROBOT HARDWARE COMMANDS ---
            elif "home" in cmd_clean:
                robot_speak("Moving to home.")
                # move_robot_home() # Place your UR RTDE code here

            elif "status" in cmd_clean:
                robot_speak("All systems nominal.")

        else:
            print("Listening...")
