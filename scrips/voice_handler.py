import os
import time
import speech_recognition as sr
import pyttsx3
from gtts import gTTS
import tempfile
import pygame

class VoiceHandler:
    def __init__(self, text_only=False):
        """Initialize voice handler component.
        
        Args:
            text_only (bool): If True, use keyboard input instead of microphone
        """
        self.text_only = text_only
        
        # Initialize speech recognition
        if not text_only:
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 4000
            self.recognizer.dynamic_energy_threshold = True
            
            # Test microphone availability
            try:
                with sr.Microphone() as source:
                    print("Initializing microphone (ambient noise calibration)...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("Microphone initialized successfully")
            except Exception as e:
                print(f"WARNING: Microphone initialization failed: {e}")
                print("Falling back to text-only mode")
                self.text_only = True
        
        # Initialize text-to-speech
        try:
            # Try pyttsx3 first (works offline)
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 175)  # Speed
            voices = self.tts_engine.getProperty('voices')
            for voice in voices:
                if "female" in voice.name.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break
            self.use_gtts = False
        except:
            # Fall back to gTTS (requires internet)
            pygame.mixer.init()
            self.use_gtts = True
            print("WARNING: Using gTTS for speech output (requires internet)")
    
    def listen(self):
        """Listen for user input and convert to text.
        
        Returns:
            str: Transcribed text from speech or direct text input
        """
        if self.text_only:
            # Text-only mode: get input from console
            return input("Type your query: ")
        
        # Voice mode: use microphone
        try:
            with sr.Microphone() as source:
                print("Listening... (Speak now)")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                print("Processing speech...")
                
                # Use Google's speech recognition
                text = self.recognizer.recognize_google(audio)
                return text
                
        except sr.WaitTimeoutError:
            print("No speech detected. Please try again.")
            return None
        except sr.UnknownValueError:
            print("Could not understand audio. Please try again.")
            return None
        except sr.RequestError as e:
            print(f"Speech service error: {e}")
            print("Falling back to text input. Please type your query:")
            return input("Type your query: ")
        except Exception as e:
            print(f"Error during speech recognition: {e}")
            return None
    
    def speak(self, text):
        """Convert text to speech.
        
        Args:
            text (str): Text to be spoken
        """
        if self.text_only:
            # In text-only mode, we don't produce speech
            return
        
        try:
            if self.use_gtts:
                # Use gTTS
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
                    temp_filename = f.name
                
                tts = gTTS(text=text, lang='en')
                tts.save(temp_filename)
                
                pygame.mixer.music.load(temp_filename)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                
                # Clean up the temporary file
                os.unlink(temp_filename)
            else:
                # Use pyttsx3
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
        except Exception as e:
            print(f"WARNING: Speech output failed: {e}")
