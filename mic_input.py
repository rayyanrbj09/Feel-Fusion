import os
import io
import wave
import pyaudio
import logging
import dotenv
from google.cloud import speech
from pynput.mouse import Listener

# Configure logging
logging.basicConfig(level=logging.INFO)

class AudioRecorder:
    def __init__(self, filename="output.wav", chunk_size=1024, format=pyaudio.paInt16, channels=1, rate=16000):
        self.filename = filename
        self.chunk_size = chunk_size
        self.format = format
        self.channels = channels
        self.rate = rate
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stop_recording = False  # Flag to stop recording

    def on_click(self, x, y, button, pressed):
        """Mouse click event handler."""
        if pressed:  # Trigger only on mouse press
            logging.info(f"Mouse clicked at ({x}, {y}). Stopping recording...")
            self.stop_recording = True
            return False  # Stop the listener

    def record_audio(self):
        """Record audio from the microphone until a mouse click is detected."""
        logging.info("Recording... Click the mouse to stop.")
        stream = self.pyaudio_instance.open(format=self.format, channels=self.channels, rate=self.rate, input=True, frames_per_buffer=self.chunk_size)
        frames = []

        # Start the mouse listener in a non-blocking way
        listener = Listener(on_click=self.on_click)
        listener.start()

        try:
            while not self.stop_recording:
                data = stream.read(self.chunk_size)
                frames.append(data)
        except Exception as e:
            logging.error(f"Error during recording: {e}")
        finally:
            listener.stop()
            stream.stop_stream()
            stream.close()
            self.pyaudio_instance.terminate()

        self.save_audio(frames)

    def save_audio(self, frames):
        """Save the recorded audio to a WAV file."""
        with wave.open(self.filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.pyaudio_instance.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
        logging.info(f"Audio saved to {self.filename}")

class AudioTranscriber:
    def __init__(self, filename="output.wav"):
        credentials_path = dotenv.get_key('.env', 'GOOGLE_APPLICATION_CREDENTIALS')
        self.filename = filename
        try:
            if credentials_path:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            self.client = speech.SpeechClient()
        except Exception as e:
            logging.error(f"Error initializing Google Cloud Speech client: {e}")
            raise

    def transcribe_audio(self):
        """Transcribe the recorded audio file using Google Cloud Speech-to-Text API."""
        logging.info("Transcribing audio...")
        try:
            with io.open(self.filename, "rb") as audio_file:
                content = audio_file.read()

            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="en-US",
                enable_automatic_punctuation=True,  # Enable punctuation
                model="video"  # Use a model optimized for video
            )

            response = self.client.long_running_recognize(config=config, audio=audio)
            response = response.result(timeout=90)  # Wait for the transcription to complete

            transcripts = [result.alternatives[0].transcript for result in response.results]
            for transcript in transcripts:
                logging.info(f"Transcript: {transcript}")
            return transcripts

        except Exception as e:
            logging.error(f"Error during transcription: {e}")
            return None

if __name__ == "__main__":
    recorder = AudioRecorder()
    recorder.record_audio()
    transcriber = AudioTranscriber()
    transcriber.transcribe_audio()