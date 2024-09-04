import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import threading

app = Flask(__name__)
socketio = SocketIO(app)

# Argument parsing
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the English model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real-time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    return parser.parse_args()

args = parse_arguments()

# Variables
phrase_time = None
data_queue = Queue()
transcription = ['']
recorder = sr.Recognizer()
recorder.energy_threshold = args.energy_threshold
recorder.dynamic_energy_threshold = False

# Initialize Microphone
if 'linux' in platform:
    mic_name = args.default_microphone
    if not mic_name or mic_name == 'list':
        print("Available microphone devices are: ")
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"Microphone with name \"{name}\" found")
        exit(0)
    else:
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            if mic_name in name:
                source = sr.Microphone(sample_rate=16000, device_index=index)
                break
else:
    source = sr.Microphone(sample_rate=16000)

# Load Whisper model
model = args.model
if args.model != "large" and not args.non_english:
    model = model + ".en"
audio_model = whisper.load_model(model)

record_timeout = args.record_timeout
phrase_timeout = args.phrase_timeout

# Flask route to serve the frontend
@app.route('/')
def index():
    return render_template('index.html')

def record_callback(_, audio: sr.AudioData) -> None:
    data = audio.get_raw_data()
    data_queue.put(data)

# Start the transcription process in a separate thread
def start_transcription():
    global phrase_time
    with source:
        recorder.adjust_for_ambient_noise(source)
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    while True:
        try:
            now = datetime.utcnow()
            if not data_queue.empty():
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                phrase_time = now

                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()

                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                text = result['text'].strip()

                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text

                # Emit the transcription to the frontend
                socketio.emit('transcription', {'data': text})

                os.system('cls' if os.name == 'nt' else 'clear')
                for line in transcription:
                    print(line)
                print('', end='', flush=True)
            else:
                sleep(0.25)
        except KeyboardInterrupt:
            break

# Run the transcription in a separate thread
threading.Thread(target=start_transcription).start()

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)
