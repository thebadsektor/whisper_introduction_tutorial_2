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

@app.route('/')
def index():
    return render_template('index.html')

# TODO#1 - Define the argument parser
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

# Parse the arguments and set configuration values
args = parse_arguments()


# TODO#2 - Set up variables for audio capture and transcription

# Set up variables to store captured audio and manage the transcription process.
phrase_time = None
data_queue = Queue()
transcription = ['']
recorder = sr.Recognizer()
recorder.energy_threshold = args.energy_threshold
recorder.dynamic_energy_threshold = False

# TODO#3 - Initialize the microphone

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

# TODO#4 - Load the Whisper model

# The Whisper model is used for transcribing the captured audio. 
model = args.model
if args.model != "large" and not args.non_english:
    model = model + ".en"
audio_model = whisper.load_model(model)

# TODO#5 - Capture audio in real-time and send data to the queue

# This function captures the audio from the microphone and puts it in a queue for transcription.
def record_callback(_, audio: sr.AudioData) -> None:
    data = audio.get_raw_data()
    data_queue.put(data)

# This function handles the real-time transcription process and emits the result to the frontend.
def start_transcription():
    
    global phrase_time
    with source:
        # TODO#6 - Adjust microphone settings to handle ambient noise and start listening for audio
        pass

    while True:
        try:
            now = datetime.utcnow()
            
            # TODO#7 - Modify the condition to check if there is data in the queue
            # Replace the True condition with a valid check to process audio data
            if not data_queue.empty():  
                # TODO#8 - Process the audio data for transcription
                # Join audio data from the queue and prepare it for transcription
                # Check if enough time has passed between phrases to complete a transcription segment
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=args.phrase_timeout):
                    phrase_complete = True
                phrase_time = now

                # Here, you will convert the raw audio into a format suitable for Whisper and perform transcription.
                # Clear the queue and process the audio data for transcription
                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()

                # TODO#9 - Transcribe the audio and send the result to the frontend via SocketIO
                # Emit the transcription result to the web interface for display
                # Convert the audio bytes to a NumPy array for processing.
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
                text = result['text'].strip()
                # Based on whether the phrase is complete, either append the text or update the last segment

                # Send the transcribed text to the frontend and update the displayed transcription
                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text

                # Send the transcribed text to the frontend in real time.
                socketio.emit('transcription', {'data': text})

                # Clear the console for real-time display of updated transcriptions
                os.system('cls' if os.name == 'nt' else 'clear')
                for line in transcription:
                    print(line)
                print('', end='', flush=True)
            else:
                # If there's no data, sleep briefly before checking again
                sleep(0.25)
        except KeyboardInterrupt:
            # Break out of the loop gracefully if interrupted
            break

# TODO#10 - Run the transcription in a separate thread
threading.Thread(target=start_transcription).start()

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)
