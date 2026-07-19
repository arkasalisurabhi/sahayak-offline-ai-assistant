import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile

SAMPLE_RATE = 16000  # Whisper and wake-word models expect 16kHz
DURATION = 5  # seconds

print("Recording for 5 seconds... speak now!")
recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()  # wait until recording is finished
print("Recording finished.")

# Save to file so you can inspect it
wavfile.write("E:/VoiceAssistant/logs/test_recording.wav", SAMPLE_RATE, recording)
print("Saved to logs/test_recording.wav")

# Play it back immediately
print("Playing back...")
sd.play(recording, SAMPLE_RATE)
sd.wait()
print("Done.")
