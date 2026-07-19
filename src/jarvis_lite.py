import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import time
import subprocess
import threading
import re
import datetime
import requests
from openwakeword.model import Model
from faster_whisper import WhisperModel
import ollama
import re as re_module

# ---------- Config ----------
SAMPLE_RATE = 16000
CHUNK_SIZE = 1280
COOLDOWN_SECONDS = 3
COMMAND_PATH = "E:/VoiceAssistant/logs/command.wav"
RESPONSE_WAV = "E:/VoiceAssistant/logs/response.wav"
PIPER_MODEL_EN = "E:/VoiceAssistant/models/piper/en_US-bryce-medium.onnx"
PIPER_MODEL_HI = "E:/VoiceAssistant/models/piper/hi_IN-pratham-medium.onnx"

# ---------- Load models once at startup ----------
print("Loading wake word model...", flush=True)
owwModel = Model(wakeword_models=["alexa"])

print("Loading Whisper model...", flush=True)
whisperModel = WhisperModel("small", device="cpu", compute_type="int8", download_root="E:/VoiceAssistant/models/whisper")

last_trigger_time = 0

# ---------- Conversation memory ----------
conversation_history = [
    {"role": "system", "content": "You are a helpful, concise voice assistant. Keep answers short - 1 to 2 sentences, spoken style, no markdown or lists."}
]
MAX_HISTORY_TURNS = 6

# ---------- Recording / Transcription ----------
def record_command(duration=5):
    print("DEBUG: entered record_command", flush=True)
    time.sleep(0.3)
    print("Listening for your command - speak now...", flush=True)
    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    print("DEBUG: recording finished", flush=True)
    wavfile.write(COMMAND_PATH, SAMPLE_RATE, recording)
    print("Command recorded.", flush=True)

def transcribe_command():
    print("DEBUG: starting transcription", flush=True)
    segments, info = whisperModel.transcribe(COMMAND_PATH, beam_size=10)
    text = "".join(segment.text for segment in segments).strip()
    detected_lang = info.language
    print(f"You said: {text} (language: {detected_lang}, confidence: {info.language_probability:.2f})", flush=True)
    
    if info.language_probability < 0.5:
        print("Low confidence transcription - likely silence/noise, ignoring.", flush=True)
        return "", detected_lang
    
    return text, detected_lang

# ---------- Skills ----------
def skill_time(user_text):
    now = datetime.datetime.now()
    return f"It's {now.strftime('%I:%M %p')} right now."

def skill_date(user_text):
    today = datetime.datetime.now()
    return f"Today is {today.strftime('%A, %B %d, %Y')}."

def skill_calculator(user_text):
    text = user_text.lower()
    text = text.replace("plus", "+").replace("minus", "-")
    text = text.replace("times", "*").replace("multiplied by", "*")
    text = text.replace("divided by", "/").replace(" x ", " * ")
    match = re.search(r'[\d\.\+\-\*/\(\)\s]+', text)
    if match:
        expression = match.group().strip()
        try:
            result = eval(expression, {"__builtins__": {}})
            return f"That's {result}."
        except Exception:
            return "Sorry, I couldn't calculate that."
    return "Sorry, I couldn't find numbers to calculate."

def skill_weather(user_text):
    try:
        lat, lon = 12.9716, 77.5946
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        response = requests.get(url, timeout=5)
        data = response.json()
        temp = data["current_weather"]["temperature"]

        is_hindi = any(c for c in user_text if '\u0900' <= c <= '\u097F')
        if is_hindi:
            return f"बैंगलोर में अभी तापमान {temp} डिग्री सेल्सियस है।"
        return f"It's currently {temp} degrees Celsius in Bangalore."
    except Exception:
        return "Sorry, I couldn't fetch the weather right now."

def route_skill(user_text):
    text = user_text.lower()

    time_words = ["time", "समय", "टाइम"]
    date_words = ["date", "तारीख", "दिनांक", "what day"]
    weather_words = ["weather", "temperature", "how hot", "how cold", "मौस", "मोस", "मोसम", "मौसम", "तापमान"]
    weather_pattern = re_module.compile(r'[मौमोमु]स्?स?म')

    if any(word in text for word in time_words):
        return skill_time(user_text)

    if any(word in text for word in date_words):
        return skill_date(user_text)

    if any(word in text for word in weather_words) or weather_pattern.search(text):
        return skill_weather(user_text)

    if any(phrase in text for phrase in ["plus", "minus", "times", "multiplied", "divided", "+", "-", "*", "/"]) and any(char.isdigit() for char in text):
        return skill_calculator(user_text)

    return None

# ---------- Conversation reset ----------
def reset_conversation():
    global conversation_history
    conversation_history = [conversation_history[0]]
    print("Conversation memory cleared.", flush=True)

# ---------- LLM ----------
def ask_llm(user_text):
    print("DEBUG: asking LLM", flush=True)
    if not user_text:
        return "Sorry, I didn't catch that."

    if user_text.strip().lower() in ["reset conversation", "forget everything", "clear memory"]:
        reset_conversation()
        return "Okay, I've cleared our conversation."

    skill_response = route_skill(user_text)
    if skill_response:
        print(f"Skill used | Response: {skill_response}", flush=True)
        conversation_history.append({"role": "user", "content": user_text})
        conversation_history.append({"role": "assistant", "content": skill_response})
        return skill_response

    conversation_history.append({"role": "user", "content": user_text})
    response = ollama.chat(model="llama3.2", messages=conversation_history)
    reply = response["message"]["content"]
    conversation_history.append({"role": "assistant", "content": reply})

    if len(conversation_history) > (MAX_HISTORY_TURNS * 2 + 1):
        conversation_history[:] = [conversation_history[0]] + conversation_history[-(MAX_HISTORY_TURNS * 2):]

    print(f"Assistant: {reply}", flush=True)
    return reply

# ---------- TTS ----------
def speak(text, lang="en"):
    print("DEBUG: starting speak()", flush=True)
    try:
        model_path = PIPER_MODEL_HI if lang == "hi" else PIPER_MODEL_EN

        if lang == "hi":
            safe_text = text.strip()
        else:
            safe_text = text.encode("ascii", "ignore").decode("ascii").strip()

        if not safe_text:
            safe_text = "Sorry, I had trouble generating a response."

        process = subprocess.run(
            ["piper", "--model", model_path, "--output_file", RESPONSE_WAV],
            input=safe_text.encode("utf-8"),
            capture_output=True
        )
        if process.returncode != 0:
            print("Piper error:", process.stderr.decode(errors="ignore"), flush=True)
            return
        rate, data = wavfile.read(RESPONSE_WAV)
        sd.play(data, rate)
        sd.wait()
        print("DEBUG: finished speaking", flush=True)
    except Exception as e:
        print("DEBUG: speak() failed safely:", e, flush=True)

# ---------- Wake word ----------
wake_detected = threading.Event()

def audio_callback(indata, frames, time_info, status):
    global last_trigger_time
    if status:
        print(status, flush=True)
    if wake_detected.is_set():
        return
    audio = np.frombuffer(indata, dtype=np.int16)
    prediction = owwModel.predict(audio)
    for mdl, score in prediction.items():
        if score > 0.4 and (time.time() - last_trigger_time) > COOLDOWN_SECONDS:
            print(f"\nWake word detected! (confidence: {score:.2f})", flush=True)
            last_trigger_time = time.time()
            wake_detected.set()

# ---------- Main loop ----------
print("\nSahayak is ready. Say 'Alexa' to start...\n", flush=True)

stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16',
                         blocksize=CHUNK_SIZE, callback=audio_callback)
stream.start()

try:
    while True:
        if wake_detected.is_set():
            try:
                record_command(duration=5)
                text, detected_lang = transcribe_command()
                reply = ask_llm(text)
                speak(reply, lang=detected_lang)
            except Exception as e:
                print("ERROR in pipeline:", e, flush=True)
            wake_detected.clear()
            print("\nListening for wake word again...\n", flush=True)
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nStopped.", flush=True)
    stream.stop()
    stream.close()