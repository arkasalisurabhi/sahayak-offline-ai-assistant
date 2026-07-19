\#  Sahayak — Offline AI Voice Assistant 



A fully offline-capable voice assistant built with wake-word detection, speech-to-text, local LLM reasoning, and text-to-speech — no cloud dependency, zero cost.



\## Features

\- Wake word activation ("Alexa") using openWakeWord

\- Speech-to-text transcription using Whisper (faster-whisper)

\- Conversational reasoning using a local LLM (Ollama / Llama 3.2)

\- Natural-sounding text-to-speech using Piper TTS

\- Fully offline pipeline — no API costs, no internet required after setup



\## Tech Stack

\- Python 3.11

\- openWakeWord — wake word detection

\- faster-whisper — speech-to-text

\- Ollama (Llama 3.2) — local LLM reasoning

\- Piper TTS — text-to-speech

\- sounddevice — audio I/O



\## Architecture

Wake Word Detection → Audio Recording → Speech-to-Text →

LLM Reasoning → Text-to-Speech → Spoken Response

\## Setup

1\. Clone this repo

2\. Create a virtual environment: `python -m venv venv`

3\. Activate it and install dependencies: `pip install -r requirements.txt`

4\. Install \[Ollama](https://ollama.com) and pull a model: `ollama pull llama3.2`

5\. Download a \[Piper voice model](https://rhasspy.github.io/piper-samples/) and place it in `models/piper/`

6\. Run: `python src/jarvis\_lite.py`



\## Status

Actively in development — Week 1 of a 3-week build. Current milestone: full end-to-end voice pipeline working (wake word → record → transcribe → reason → speak).



\## Roadmap

\- \[x] Wake word detection

\- \[x] Speech-to-text integration

\- \[x] LLM reasoning integration

\- \[x] Text-to-speech integration

\- \[ ] Conversation memory (multi-turn context)

\- \[ ] Custom skills (time, weather, timers)

\- \[ ] Multilingual support

\- \[ ] Packaged desktop app



\## Author

Surabhi — IoT Engineering student, MVJ College of Engineering

