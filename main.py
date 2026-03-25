import subprocess
import os
import sounddevice as sd
import soundfile as sf
import time

# --- CONFIGURATION ---
# Ensure these files are in the same folder as this script
PIPER_EXE = "piper/piper.exe"
MODEL_PATH = "models/en_US-amy-medium.onnx"
TEMP_WAV = "last_speech.wav"

def robot_speak(text, speed=0.75):
    """
    speed: 1.0 is default. Lower is faster (0.8 is recommended).
    """
    if not os.path.exists(PIPER_EXE):
        print("ERROR: piper.exe missing!")
        return

    try:
        command = [
            PIPER_EXE,
            "--model", MODEL_PATH,
            "--length_scale", str(speed),  # This is the speed control
            "--output_file", TEMP_WAV
        ]

        process = subprocess.Popen(
            command, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )

        process.communicate(input=text)

        if os.path.exists(TEMP_WAV):
            data, fs = sf.read(TEMP_WAV, dtype='float32')
            sd.play(data, fs)
            sd.wait()
            os.remove(TEMP_WAV)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # --- TEST SUITE ---
    print("--- UR3e Voice System Initialization ---")
    
    # Test 1: Simple greeting
    robot_speak("System online. All safety protocols active.")
    
    time.sleep(1) # Small pause
    
    # Test 2: Complex sentence to verify phonemes
    robot_speak("The voice engine is now fully operational. No more errors detected.")
