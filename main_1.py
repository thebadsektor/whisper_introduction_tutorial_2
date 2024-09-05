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

# Flask route to serve the frontend
@app.route('/')
def index():
    return render_template('index.html')

data_queue = Queue()
# TODO#1 - Initialize Variables for Transcription and Recorder
# Variables
# TODO#2 - Set Up and Initialize the Microphone
# TODO#3 - Load the Whisper Model for Audio Transcription
# TODO#4 - Set Recording and Phrase Timeout Parameters
def start_transcription():
    # TODO#5 - Adjust Microphone Settings and Begin Listening
    while True:
        try:
            now = datetime.utcnow()
            if not data_queue.empty():
                # TODO#6 - Check and Process the Audio Data in Real-Time
                # TODO#7 - Convert the Audio Data for Transcription
                # TODO#8 - Emit Transcription Results to the Frontend Using SocketIO
                pass
            else:
                sleep(0.25)
        except KeyboardInterrupt:
            break

# TODO#9 - Start the Transcription in a Separate Thread

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)