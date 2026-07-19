import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import time
import subprocess
from openwakeword.model import Model
from faster_whisper import WhisperModel
import ollama
import threading

# ---------- Config ----------
SAMPLE_RATE = 16000
CHUNK_SIZE = 1280
COOLDOWN_SECONDS = 3
COMMAND_PATH = "E:/VoiceAssistant/logs/command.wav"
RESPONSE_WAV = "E:/VoiceAssistant/logs/response.wav"
PIPER_MODEL = "E:/VoiceAssistant/models/piper/en_US-bryce-medium.onnx"

# ---------- Load models once at startup ----------
print("Loading wake word model...", flush=True)
owwModel = Model(wakeword_models=["alexa"])

print("Loading Whisper model...", flush=True)
whisperModel = WhisperModel("base", device="cpu", compute_type="int8",
                             download_root="E:/VoiceAssistant/models/whisper")

last_trigger_time = 0

# ---------- Conversation memory ----------
conversation_history = [
    {"role": "system", "content": "You are a helpful, concise voice assistant. Keep answers short — 1 to 2 sentences, spoken style, no markdown or lists."}
]
MAX_HISTORY_TURNS = 6  # keep last 6 exchanges (12 messages) to avoid unbounded growth

# ---------- Functions ----------
def record_command(duration=5):
    print("DEBUG: entered record_command", flush=True)
    time.sleep(0.3)
    print("🎤 Listening for your command — speak now...", flush=True)
    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    print("DEBUG: recording finished", flush=True)
    wavfile.write(COMMAND_PATH, SAMPLE_RATE, recording)
    print("✅ Command recorded.", flush=True)

def transcribe_command():
    print("DEBUG: starting transcription", flush=True)
    segments, info = whisperModel.transcribe(COMMAND_PATH, beam_size=5)
    text = "".join(segment.text for segment in segments).strip()
    print(f"📝 You said: {text}", flush=True)
    return text

def ask_llm(user_text):
    print("DEBUG: asking LLM", flush=True)
    if not user_text:
        return "Sorry, I didn't catch that."

    # Handle reset command
    if user_text.strip().lower() in ["reset conversation", "forget everything", "clear memory"]:
        reset_conversation()
        return "Okay, I've cleared our conversation."

    conversation_history.append({"role": "user", "content": user_text})

    response = ollama.chat(
        model="llama3.2",
        messages=conversation_history
    )
    reply = response["message"]["content"]
    conversation_history.append({"role": "assistant", "content": reply})

    # Trim history to avoid unbounded growth (keep system prompt + last N turns)
    if len(conversation_history) > (MAX_HISTORY_TURNS * 2 + 1):
        conversation_history[:] = [conversation_history[0]] + conversation_history[-(MAX_HISTORY_TURNS * 2):]

    print(f"🤖 Assistant: {reply}", flush=True)
    return reply

def reset_conversation():
    global conversation_history
    conversation_history = [conversation_history[0]]  # keep only the system prompt
    print("🔄 Conversation memory cleared.", flush=True)
    
def speak(text):
    print("DEBUG: starting speak()", flush=True)
    process = subprocess.run(
        ["piper", "--model", PIPER_MODEL, "--output_file", RESPONSE_WAV],
        input=text.encode("utf-8"),
        capture_output=True
    )
    if process.returncode != 0:
        print("Piper error:", process.stderr.decode(), flush=True)
        return
    rate, data = wavfile.read(RESPONSE_WAV)
    sd.play(data, rate)
    sd.wait()
    print("DEBUG: finished speaking", flush=True)

wake_detected = threading.Event()

def audio_callback(indata, frames, time_info, status):
    global last_trigger_time
    if status:
        print(status, flush=True)
    if wake_detected.is_set():
        return  # already triggered, ignore further audio until reset
    audio = np.frombuffer(indata, dtype=np.int16)
    prediction = owwModel.predict(audio)
    for mdl, score in prediction.items():
        if score > 0.5 and (time.time() - last_trigger_time) > COOLDOWN_SECONDS:
            print(f"\n✅ Wake word detected! (confidence: {score:.2f})", flush=True)
            last_trigger_time = time.time()
            wake_detected.set()

# ---------- Main loop ----------
print("\n🟢 Jarvis-Lite is ready. Say 'Alexa' to start...\n", flush=True)

stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16',
                         blocksize=CHUNK_SIZE, callback=audio_callback)
stream.start()

try:
    while True:
        if wake_detected.is_set():
            try:
                record_command(duration=5)
                text = transcribe_command()
                reply = ask_llm(text)
                speak(reply)
            except Exception as e:
                print("❌ ERROR in pipeline:", e, flush=True)
            wake_detected.clear()
            print("\n🟢 Listening for wake word again...\n", flush=True)
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nStopped.", flush=True)
    stream.stop()
    stream.close()