import subprocess
import os
import sys
import sounddevice as sd
import soundfile as sf
import time
import pyaudio
import numpy as np
from faster_whisper import WhisperModel

# --- CONFIGURATION ---
print("--- UR3e Voice System Initialization ---")

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

def listen_for_command():
    chunk = 1024
    fs = 16000
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True, frames_per_buffer=chunk)
        print("\n👂 Listening...")
        
        frames = []
        # Reduced to ~2.5 seconds for faster response
        for _ in range(0, int(fs / chunk * 2.5)):
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        
        # Convert to float32
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0
        
        # SPEED HACK: beam_size=1 and vad_filter=True
        segments, _ = model.transcribe(
            audio_data, 
            beam_size=1,        # Don't overthink it (HUGE speed boost)
            vad_filter=True,    # Filter out silence/noise automatically
            language="en"       # Force English so it doesn't spend time guessing the language
        )
        
        return "".join([s.text for s in segments]).strip()

    except Exception as e:
        print(f"STT Error: {e}")
        return ""
    finally:
        p.terminate()

    # Process audio
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0
    segments, _ = model.transcribe(audio_data, beam_size=5)
    
    return "".join([segment.text for segment in segments]).strip()

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
        command = listen_for_command()
        
        if command:
            print(f"User Input: {command}")
            cmd_clean = command.lower()

            # --- TERMINATION COMMAND ---
            if "terminate program" in cmd_clean or "stop all systems" in cmd_clean:
                print("!!! TERMINATION SIGNAL RECEIVED !!!")
                robot_speak("Terminating all systems. Goodbye.")
                # We wait a split second for the speaker to finish before killing the process
                time.sleep(0.5) 
                sys.exit() # This kills the entire Python process immediately

            # --- OTHER ROBOT COMMANDS ---
            elif "home" in cmd_clean:
                robot_speak("Moving to home.")
                # [Insert UR3e MoveHome code here]

            elif "status" in cmd_clean:
                robot_speak("All systems nominal.")

        else:
            # This keeps the console alive so you know it didn't crash
            print("Listening...")