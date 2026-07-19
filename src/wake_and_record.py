import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import time
from openwakeword.model import Model

SAMPLE_RATE = 16000
CHUNK_SIZE = 1280
RECORD_DURATION = 5
COOLDOWN_SECONDS = 3  # prevent multiple triggers from one utterance

owwModel = Model(wakeword_models=["alexa"])
last_trigger_time = 0

def record_command(duration=5):
    time.sleep(0.3)
    print("🎤 Listening for your command NOW — speak clearly...")
    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    wavfile.write("E:/VoiceAssistant/logs/command.wav", SAMPLE_RATE, recording)
    print("✅ Command recorded and saved.")
   

def audio_callback(indata, frames, time_info, status):
    global last_trigger_time
    audio = np.frombuffer(indata, dtype=np.int16)
    prediction = owwModel.predict(audio)

    for mdl, score in prediction.items():
        if score > 0.5:
            now = time.time()
            if now - last_trigger_time > COOLDOWN_SECONDS:
                last_trigger_time = now
                print(f"Wake word detected! (confidence: {score:.2f})")
                record_command()
                print("Listening for wake word 'alexa' again...\n")

print("Listening for wake word 'alexa'... (Ctrl+C to stop)")
with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16',
                     blocksize=CHUNK_SIZE, callback=audio_callback):
    try:
        while True:
            sd.sleep(100)
    except KeyboardInterrupt:
        print("\nStopped listening.")