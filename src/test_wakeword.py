import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import time
from openwakeword.model import Model

owwModel = Model(wakeword_models=["alexa"])

SAMPLE_RATE = 16000
CHUNK_SIZE = 1280
RECORD_SECONDS = 5

def record_command():
    print("🎙️ Listening for your command...")
    recording = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    filename = "E:/VoiceAssistant/logs/command.wav"
    wavfile.write(filename, SAMPLE_RATE, recording)
    print(f"✅ Command recorded and saved to {filename}\n")

print("Listening for wake word 'Alexa'... (Ctrl+C to stop)\n")

try:
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', blocksize=CHUNK_SIZE) as stream:
        while True:
            audio_chunk, _ = stream.read(CHUNK_SIZE)
            audio = audio_chunk.flatten()
            prediction = owwModel.predict(audio)

            triggered = False
            for mdl, score in prediction.items():
                if score > 0.5:
                    print(f"✅ Wake word detected! (confidence: {score:.2f})")
                    triggered = True
                    break

            if triggered:
                record_command()
                # Clear prediction buffer/cooldown by sleeping briefly
                time.sleep(1)
                print("Listening for wake word 'Alexa' again...\n")

except KeyboardInterrupt:
    print("\nStopped listening.")
