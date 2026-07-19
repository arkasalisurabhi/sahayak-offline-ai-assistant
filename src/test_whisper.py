from faster_whisper import WhisperModel

# Model loads once; "base" is a good balance of speed/accuracy for CPU
model = WhisperModel("base", device="cpu", compute_type="int8",
                      download_root="E:/VoiceAssistant/models/whisper")

audio_path ="E:/VoiceAssistant/logs/command.wav"

print("Transcribing...")
segments, info = model.transcribe(audio_path, beam_size=5, language="en")

print(f"Detected language: {info.language} (confidence: {info.language_probability:.2f})")

full_text = ""
for segment in segments:
    full_text += segment.text

print("Transcription:", full_text.strip())