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

# TODO#2 - Set up variables for audio capture and transcription

# TODO#3 - Initialize the microphone

# TODO#4 - Load the Whisper model

# TODO#5 - Capture audio in real-time and send data to the queue

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
            if True:  
                # TODO#8 - Process the audio data for transcription
                # Join audio data from the queue and prepare it for transcription

                # TODO#9 - Transcribe the audio and send the result to the frontend via SocketIO
                # Emit the transcription result to the web interface for display

                pass
            else:
                # If there's no data, sleep briefly before checking again
                sleep(0.25)
        except KeyboardInterrupt:
            # Break out of the loop gracefully if interrupted
            break

# TODO#10 - Run the transcription in a separate thread

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)
