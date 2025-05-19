"""
# DEVELOPMENT OF A MODEL FOR CUSTOMER SENTIMENT ANALYSIS IN VOICE-ENABLED E-COMMERCE

This project implements a voice-enabled e-commerce system that can analyze customer sentiment
through spoken interactions. The system transcribes voice input, analyzes the sentiment,
and responds appropriately based on detected customer emotions.

## Project Structure

- `main.py`: Core application entry point
- `voice_handler.py`: Voice input/output functionality
- `sentiment_analyzer.py`: Sentiment analysis model
- `ecommerce_system.py`: Mock e-commerce system
- `utils.py`: Helper functions and utilities
- `requirements.txt`: Required dependencies

## Setup and Installation Guide
"""

# Let's start with requirements.txt file content
import textwrap

requirements = """
# Core libraries
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2

# Voice processing
SpeechRecognition==3.10.0
PyAudio==0.2.13
pyttsx3==2.90
gTTS==2.3.2

# NLP and Sentiment Analysis
nltk==3.8.1
transformers==4.30.2
torch==2.0.1
scikit-learn==1.3.0

# Utilities
tqdm==4.65.0
python-dotenv==1.0.0
"""

# main.py - Core application
main_py = """
"""

main_py = """
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DEVELOPMENT OF A MODEL FOR CUSTOMER SENTIMENT ANALYSIS IN VOICE-ENABLED E-COMMERCE

This is the main application entry point that brings together all components:
- Voice input/output handling
- Sentiment analysis
- E-commerce system interaction
"""

import os
import time
import argparse
from voice_handler import VoiceHandler
from sentiment_analyzer import SentimentAnalyzer
from ecommerce_system import EcommerceSystem
import utils

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Voice-Enabled E-commerce Sentiment Analysis')
    parser.add_argument('--no-voice', action='store_true', help='Run in text-only mode (no voice)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    return parser.parse_args()

def main():
    """Main application entry point."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize components
    print("Initializing system components...")
    voice_handler = VoiceHandler(text_only=args.no_voice)
    sentiment_analyzer = SentimentAnalyzer()
    ecommerce = EcommerceSystem()
    
    # Application state
    session_active = True
    customer_name = None
    shopping_cart = []
    sentiment_history = []
    
    # Welcome message
    welcome_message = "Welcome to our voice-enabled e-commerce system! How can I help you today?"
    print("\n" + "-"*80)
    print("SYSTEM: " + welcome_message)
    voice_handler.speak(welcome_message)
    
    # Main interaction loop
    while session_active:
        # Get user input
        print("\n" + "-"*40)
        user_input = voice_handler.listen()
        if not user_input:
            continue
            
        print(f"YOU: {user_input}")
        
        # Process commands
        if "exit" in user_input.lower() or "quit" in user_input.lower() or "goodbye" in user_input.lower():
            farewell = utils.generate_farewell(sentiment_history)
            print("SYSTEM: " + farewell)
            voice_handler.speak(farewell)
            session_active = False
            continue
        
        # Analyze sentiment
        sentiment = sentiment_analyzer.analyze(user_input)
        sentiment_score = sentiment['compound']
        sentiment_category = sentiment['category']
        sentiment_history.append(sentiment)
        
        if args.debug:
            print(f"DEBUG - Sentiment: {sentiment_category} (Score: {sentiment_score:.2f})")
        
        # Process e-commerce interactions
        response = ecommerce.process_query(user_input, sentiment)
        
        # Add sentiment-aware elements to response
        response = utils.enhance_response_with_sentiment(response, sentiment)
        
        # Output response
        print("SYSTEM: " + response)
        voice_handler.speak(response)
    
    # Session summary
    if len(sentiment_history) > 1:
        avg_sentiment = sum(s['compound'] for s in sentiment_history) / len(sentiment_history)
        print("\n" + "="*80)
        print(f"Session Summary:")
        print(f"Total interactions: {len(sentiment_history)}")
        print(f"Average sentiment: {avg_sentiment:.2f} ({utils.score_to_category(avg_sentiment)})")
        print("="*80)
    
    print("\nThank you for using our voice-enabled e-commerce system!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram terminated by user.")
    except Exception as e:
        print(f"\n\nAn error occurred: {e}")
"""

# voice_handler.py
voice_handler_py = """
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Voice input/output handling component for the e-commerce sentiment analysis system.
This module provides speech-to-text and text-to-speech functionality.
"""

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
"""

# sentiment_analyzer.py
sentiment_analyzer_py = """
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sentiment analysis component for the e-commerce system.
This module analyzes customer input text to determine sentiment polarity and intensity.
"""

import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch

class SentimentAnalyzer:
    def __init__(self, use_transformers=True):
        """Initialize the sentiment analyzer.
        
        Args:
            use_transformers (bool): Whether to use the Transformers model (better but slower)
                                    or just VADER (faster but less accurate)
        """
        # Download VADER lexicon if not already downloaded
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            print("Downloading NLTK resources...")
            nltk.download('vader_lexicon')
        
        # Initialize VADER sentiment analyzer
        self.vader = SentimentIntensityAnalyzer()
        
        # Initialize Transformers model if requested
        self.use_transformers = use_transformers
        if use_transformers:
            try:
                # Try to load the more advanced sentiment model
                model_name = "distilbert-base-uncased-finetuned-sst-2-english"
                print(f"Loading sentiment model: {model_name}...")
                
                # Load model and tokenizer
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.transformer_pipeline = pipeline(
                    "sentiment-analysis", 
                    model=self.model, 
                    tokenizer=self.tokenizer,
                    device=0 if torch.cuda.is_available() else -1
                )
                print("Advanced sentiment model loaded successfully")
            except Exception as e:
                print(f"WARNING: Could not load transformer model: {e}")
                print("Falling back to VADER sentiment analyzer only")
                self.use_transformers = False
    
    def preprocess_text(self, text):
        """Preprocess text for sentiment analysis.
        
        Args:
            text (str): The input text
            
        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze(self, text):
        """Analyze the sentiment of the given text.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            dict: Sentiment analysis results
        """
        if not text:
            return {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0, 'category': 'neutral'}
        
        # Preprocess the text
        preprocessed_text = self.preprocess_text(text)
        
        # Get VADER sentiment
        vader_scores = self.vader.polarity_scores(preprocessed_text)
        
        if self.use_transformers:
            try:
                # Get transformer model sentiment
                transformer_result = self.transformer_pipeline(preprocessed_text)[0]
                
                # Convert transformer output to a score between -1 and 1
                if transformer_result['label'] == 'POSITIVE':
                    transformer_score = transformer_result['score']
                else:
                    transformer_score = -transformer_result['score']
                
                # Blend VADER and transformer scores (weighted average)
                compound = (vader_scores['compound'] + transformer_score) / 2
                
                # Adjust other scores based on the blended compound
                if compound >= 0:
                    pos = abs(compound)
                    neg = 0.0
                    neu = 1.0 - pos
                else:
                    neg = abs(compound)
                    pos = 0.0
                    neu = 1.0 - neg
            except Exception as e:
                # Fall back to VADER if transformer fails
                print(f"WARNING: Transformer model failed: {e}")
                compound = vader_scores['compound']
                pos = vader_scores['pos']
                neg = vader_scores['neg']
                neu = vader_scores['neu']
        else:
            # Just use VADER
            compound = vader_scores['compound']
            pos = vader_scores['pos']
            neg = vader_scores['neg']
            neu = vader_scores['neu']
        
        # Determine sentiment category
        if compound >= 0.5:
            category = 'very positive'
        elif compound >= 0.05:
            category = 'positive'
        elif compound <= -0.5:
            category = 'very negative'
        elif compound <= -0.05:
            category = 'negative'
        else:
            category = 'neutral'
        
        # Detect specific emotions (basic implementation)
        emotions = []
        
        # Check for frustration indicators
        frustration_keywords = ['frustrated', 'annoying', 'difficult', 'confusing', 'waste', 'useless']
        if any(keyword in preprocessed_text for keyword in frustration_keywords):
            emotions.append('frustrated')
        
        # Check for satisfaction indicators
        satisfaction_keywords = ['great', 'wonderful', 'excellent', 'amazing', 'perfect', 'satisfied']
        if any(keyword in preprocessed_text for keyword in satisfaction_keywords):
            emotions.append('satisfied')
        
        # Return results
        return {
            'compound': compound,
            'pos': pos,
            'neg': neg,
            'neu': neu,
            'category': category,
            'emotions': emotions
        }

# Simple test if run directly
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    test_texts = [
        "I love this product, it's amazing!",
        "This is the worst experience I've ever had.",
        "The delivery was on time, but the packaging was damaged.",
        "I'm not sure if I like this or not.",
        "This is absolutely fantastic and exceeded my expectations!"
    ]
    
    for text in test_texts:
        sentiment = analyzer.analyze(text)
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment['category']} (Score: {sentiment['compound']:.2f})")
        print("-" * 50)
"""

# ecommerce_system.py
ecommerce_system_py = """
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
E-commerce system component that handles product information, user queries,
and shopping functionality.
"""

import re
import random
import json
import os

class EcommerceSystem:
    def __init__(self):
        """Initialize the e-commerce system with product data."""
        # Load product catalog
        self.products = self._load_product_catalog()
        
        # Initialize user session
        self.shopping_cart = []
        self.recently_viewed = []
        self.current_category = None
        
        # Intent patterns
        self.intent_patterns = {
            'search': r'(?:find|search|looking for|show me|do you have|searching for|need to find)\s+(.+)',
            'info': r'(?:tell me about|details on|more information about|specs for|features of|describe)\s+(.+)',
            'price': r'(?:how much|price of|cost of|what is the price of|how expensive is)\s+(.+)',
            'add_to_cart': r'(?:add|put|place)\s+(.+?)\s+(?:to|in|into|on)(?:\s+my)?\s+(?:cart|basket)',
            'remove_from_cart': r'(?:remove|take out|delete)\s+(.+?)\s+(?:from)(?:\s+my)?\s+(?:cart|basket)',
            'view_cart': r'(?:view|show|what\'s in|display|see)(?:\s+my)?\s+(?:cart|basket)',
            'checkout': r'(?:checkout|proceed to checkout|buy now|purchase|complete purchase|place order)',
            'help': r'(?:help|assist|support|how do I|how to|what can you do)',
            'recommendations': r'(?:recommend|suggestion|what do you recommend|popular items)',
            'greeting': r'(?:hi|hello|hey|greetings)',
            'feedback': r'(?:feedback|review|rate|comment)'
        }
    
    def _load_product_catalog(self):
        """Load the product catalog from file or initialize with demo data."""
        # Check if product data file exists
        if os.path.exists('product_catalog.json'):
            try:
                with open('product_catalog.json', 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading product catalog: {e}")
        
        # Create demo product catalog
        print("Creating demo product catalog...")
        categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Books', 'Beauty']
        
        products = []
        
        # Electronics
        products.extend([
            {
                'id': 'e1',
                'name': 'Smartphone XYZ',
                'category': 'Electronics',
                'price': 699.99,
                'description': 'Latest smartphone with 6.7-inch display, 128GB storage, and triple camera system.',
                'rating': 4.5,
                'stock': 25
            },
            {
                'id': 'e2',
                'name': 'Wireless Headphones',
                'category': 'Electronics',
                'price': 149.99,
                'description': 'Premium noise-cancelling wireless headphones with 30-hour battery life.',
                'rating': 4.7,
                'stock': 40
            },
            {
                'id': 'e3',
                'name': 'Smartwatch Pro',
                'category': 'Electronics',
                'price': 249.99,
                'description': 'Advanced smartwatch with health monitoring, GPS, and waterproof design.',
                'rating': 4.3,
                'stock': 15
            }
        ])
        
        # Clothing
        products.extend([
            {
                'id': 'c1',
                'name': 'Casual T-shirt',
                'category': 'Clothing',
                'price': 19.99,
                'description': 'Comfortable cotton t-shirt available in multiple colors.',
                'rating': 4.2,
                'stock': 100
            },
            {
                'id': 'c2',
                'name': 'Denim Jeans',
                'category': 'Clothing',
                'price': 59.99,
                'description': 'Classic denim jeans with straight fit.',
                'rating': 4.4,
                'stock': 75
            }
        ])
        
        # Home & Kitchen
        products.extend([
            {
                'id': 'h1',
                'name': 'Coffee Maker',
                'category': 'Home & Kitchen',
                'price': 89.99,
                'description': 'Programmable coffee maker with 12-cup capacity.',
                'rating': 4.1,
                'stock': 30
            },
            {
                'id': 'h2',
                'name': 'Non-stick Cookware Set',
                'category': 'Home & Kitchen',
                'price': 129.99,
                'description': '10-piece non-stick cookware set with glass lids.',
                'rating': 4.6,
                'stock': 20
            }
        ])
        
        # Books
        products.extend([
            {
                'id': 'b1',
                'name': 'Artificial Intelligence Basics',
                'category': 'Books',
                'price': 29.99,
                'description': 'Introduction to artificial intelligence and machine learning concepts.',
                'rating': 4.8,
                'stock': 50
            },
            {
                'id': 'b2',
                'name': 'The Bestseller Novel',
                'category': 'Books',
                'price': 14.99,
                'description': 'Award-winning fiction novel with over 1 million copies sold.',
                'rating': 4.9,
                'stock': 60
            }
        ])
        
        # Beauty
        products.extend([
            {
                'id': 'be1',
                'name': 'Face Serum',
                'category': 'Beauty',
                'price': 24.99,
                'description': 'Hydrating face serum with vitamin C for all skin types.',
                'rating': 4.5,
                'stock': 45
            },
            {
                'id': 'be2',
                'name': 'Makeup Set',
                'category': 'Beauty',
                'price': 39.99,
                'description': 'Complete makeup set with eyeshadow, lipstick, and mascara.',
                'rating': 4.3,
                'stock': 35
            }
        ])
        
        # Save the product catalog to file
        try:
            with open('product_catalog.json', 'w') as f:
                json.dump(products, f, indent=2)
        except Exception as e:
            print(f"Error saving product catalog: {e}")
        
        return products
    
    def process_query(self, query, sentiment=None):
        """Process a user query and return appropriate response.
        
        Args:
            query (str): User query text
            sentiment (dict, optional): Sentiment analysis results
            
        Returns:
            str: Response to the user query
        """
        # Check for intents
        for intent, pattern in self.intent_patterns.items():
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                # Extract entity if present
                entity = match.group(1) if len(match.groups()) > 0 else None
                
                # Process based on intent
                if intent == 'search':
                    return self._handle_search(entity)
                elif intent == 'info':
                    return self._handle_info(entity)
                elif intent == 'price':
                    return self._handle_price(entity)
                elif intent == 'add_to_cart':
                    return self._handle_add_to_cart(entity)
                elif intent == 'remove_from_cart':
                    return self._handle_remove_from_cart(entity)
                elif intent == 'view_cart':
                    return self._handle_view_cart()
                elif intent == 'checkout':
                    return self._handle_checkout()
                elif intent == 'help':
                    return self._handle_help()
                elif intent == 'recommendations':
                    return self._handle_recommendations()
                elif intent == 'greeting':
                    return self._handle_greeting()
                elif intent == 'feedback':
                    return self._handle_feedback()
        
        # If no specific intent is detected
        return self._handle_general_query(query, sentiment)
    
    def _find_product(self, product_name):
        """Find a product by name or partial name match.
        
        Args:
            product_name (str): Product name to search for
            
        Returns:
            dict: The matching product or None if not found
        """
        if not product_name:
            return None
            
        product_name = product_name.lower()
        
        # Try exact match first
        for product in self.products:
            if product['name'].lower() == product_name:
                return product
        
        # Try partial match
        for product in self.products:
            if product_name in product['name'].lower():
                return product
        
        return None
    
    def _handle_search(self, query):
        """Handle product search queries.
        
        Args:
            query (str): Search query
            
        Returns:
            str: Search results
        """
        if not query:
            return "What product are you looking for?"
            
        query = query.lower()
        results = []
        
        # Check if searching by category
        categories = set(p['category'] for p in self.products)
        matching_categories = [c for c in categories if query in c.lower()]
        
        if matching_categories:
            category = matching_categories[0]
            self.current_category = category
            category_products = [p for p in self.products if p['category'] == category]
            
            if category_products:
                product_list = ", ".join([p['name'] for p in category_products[:5]])
                if len(category_products) > 5:
                    product_list += f", and {len(category_products) - 5} more"
                
                return f"I found {len(category_products)} products in {category}: {product_list}. Would you like more information about any of these products?"
            else:
                return f"I don't have any products in the {category} category at the moment."
        
        # Search by product name
        for product in self.products:
            if query in product['name'].lower() or query in product['description'].lower():
                results.append(product)
        
        if results:
            if len(results) == 1:
                product = results[0]
                self.recently_viewed.append(product)
                return f"I found {product['name']} for ${product['price']:.2f}. {product['description']} Would you like to add this to your cart?"
            else:
                names = ", ".join([p['name'] for p in results[:5]])
                if len(results) > 5:
                    names += f", and {len(results) - 5} more"
                return f"I found {len(results)} products matching '{query}': {names}. Which one would you like to know more about?"
        else:
            return f"I couldn't find any products matching '{query}'. Would you like to see our featured products instead?"
    
    def _handle_info(self, product_name):
        """Handle requests for product information.
        
        Args:
            product_name (str): Name of product to get info for
            
        Returns:
            str: Product information
        """
        product = self._find_product(product_name)
        
        if product:
            self.recently_viewed.append(product)
            return (f"Here's information about {product['name']}: {product['description']} "
                   f"It costs ${product['price']:.2f} and has a rating of {product['rating']}/5.0. "
                   f"Currently {product['stock']} in stock. Would you like to add this to your cart?")
        else:
            return f"I couldn't find information about '{product_name}'. Could you be more specific about which product you're interested in?"
    
    def _handle_price(self, product_name):
        """Handle price inquiries.
        
        Args:
            product_name (str): Name of product to get price for
            
        Returns:
            str: Price information
        """
        product = self._find_product(product_name)
        
        if product:
            return f"The {product['name']} is priced at ${product['price']:.2f}."
        else:
            return f"I couldn't find a price for '{product_name}'. Could you specify which product you're asking about?"
    
    def _handle_add_to_cart(self, product_name):
        """Handle adding products to cart.
        
        Args:
            product_name (str): Name of product to add
            
        Returns:
            str: Confirmation message
        """
        product = self._find_product(product_name)
        
        if product:
            # Check if already in cart
            for item in self.shopping_cart:
                if item['product']['id'] == product['id']:
                    item['quantity'] += 1
                    total = sum(item['product']['price'] * item['quantity'] for item in self.shopping_cart)
                    return f"Added another {product['name']} to your cart. You now have {item['quantity']} in your cart. Your cart total is ${total:.2f}."
                else:
                    del self.shopping_cart[i]
                    total = sum(item['product']['price'] * item['quantity'] for item in self.shopping_cart)
                    return f"Removed {product['name']} from your cart. Your cart total is ${total:.2f}."
        
        return f"I couldn't find '{product_name}' in your cart."
    
    def _handle_view_cart(self):
        """Handle requests to view shopping cart.
        
        Returns:
            str: Shopping cart contents
        """
        if not self.shopping_cart:
            return "Your shopping cart is empty. Would you like to see our featured products?"
        
        total = sum(item['product']['price'] * item['quantity'] for item in self.shopping_cart)
        
        if len(self.shopping_cart) == 1:
            item = self.shopping_cart[0]
            return f"You have {item['quantity']} {item['product']['name']} in your cart, totaling ${total:.2f}. Would you like to proceed to checkout?"
        else:
            cart_items = ", ".join([f"{item['quantity']} {item['product']['name']}" for item in self.shopping_cart])
            return f"Your cart contains: {cart_items}. Total: ${total:.2f}. Would you like to proceed to checkout or continue shopping?"
    
    def _handle_checkout(self):
        """Handle checkout requests.
        
        Returns:
            str: Checkout confirmation
        """
        if not self.shopping_cart:
            return "Your shopping cart is empty. Please add some products before checking out."
        
        total = sum(item['product']['price'] * item['quantity'] for item in self.shopping_cart)
        
        # Simulate checkout process
        self.shopping_cart = []
        
        return f"Thank you for your purchase of ${total:.2f}! Your order has been confirmed and will be processed shortly. A confirmation email with tracking information will be sent to you. Is there anything else I can help you with today?"
    
    def _handle_help(self):
        """Handle help requests.
        
        Returns:
            str: Help information
        """
        return ("I can help you with the following:\n"
                "- Search for products by saying 'Find smartphones' or 'Show me kitchen appliances'\n"
                "- Get product details with 'Tell me about the Wireless Headphones'\n"
                "- Check prices with 'How much is the Coffee Maker?'\n"
                "- Add products to your cart with 'Add Smartwatch to my cart'\n"
                "- View your cart by saying 'Show my cart'\n"
                "- Remove items with 'Remove Denim Jeans from my cart'\n"
                "- Checkout by saying 'Proceed to checkout'\n"
                "- Get product recommendations by asking 'What do you recommend?'\n"
                "What would you like to do?")
    
    def _handle_recommendations(self):
        """Handle recommendation requests.
        
        Returns:
            str: Product recommendations
        """
        # Base recommendations on recently viewed items if available
        if self.recently_viewed:
            last_viewed = self.recently_viewed[-1]
            category = last_viewed['category']
            
            # Recommend products from same category but not the same product
            recommendations = [p for p in self.products if p['category'] == category and p['id'] != last_viewed['id']]
            
            if recommendations:
                recommendations.sort(key=lambda x: x['rating'], reverse=True)
                rec_products = recommendations[:3]
                rec_text = ", ".join([f"{p['name']} (${p['price']:.2f})" for p in rec_products])
                return f"Based on your interest in {last_viewed['name']}, you might also like these {category} products: {rec_text}. Would you like more details on any of these?"
        
        # Default recommendations (top-rated products)
        top_products = sorted(self.products, key=lambda x: x['rating'], reverse=True)[:3]
        rec_text = ", ".join([f"{p['name']} (${p['price']:.2f})" for p in top_products])
        return f"Our top-rated products are: {rec_text}. Would you like more information about any of these?"
    
    def _handle_greeting(self):
        """Handle greeting messages.
        
        Returns:
            str: Greeting response
        """
        greetings = [
            "Hello! Welcome to our voice-enabled shopping assistant. How can I help you today?",
            "Hi there! I'm here to help with your shopping. What are you looking for today?",
            "Welcome! I'm your virtual shopping assistant. What can I help you find today?"
        ]
        return random.choice(greetings)
    
    def _handle_feedback(self):
        """Handle feedback requests.
        
        Returns:
            str: Feedback response
        """
        return "I'd love to hear your feedback! Please share your thoughts on your shopping experience, and I'll make sure to pass it along to our team. Your input helps us improve our service."
    
    def _handle_general_query(self, query, sentiment=None):
        """Handle general queries that don't match specific intents.
        
        Args:
            query (str): User query
            sentiment (dict, optional): Sentiment analysis results
            
        Returns:
            str: Response to general query
        """
        # Check for product mentions in the query
        for product in self.products:
            if product['name'].lower() in query.lower():
                return self._handle_info(product['name'])
        
        # Check for category mentions
        categories = set(p['category'] for p in self.products)
        for category in categories:
            if category.lower() in query.lower():
                return self._handle_search(category)
        
        # Default responses based on sentiment if available
        if sentiment:
            if sentiment['category'] == 'very negative':
                return "I'm sorry you seem frustrated. How can I better assist you with your shopping today? Would you like me to show you our most popular products or help you find something specific?"
            elif sentiment['category'] == 'negative':
                return "I understand you might not be satisfied. How can I improve your shopping experience? Perhaps I can help you find a specific product or category?"
            elif sentiment['category'] == 'very positive':
                return "I'm glad you're enjoying the experience! Is there anything specific you're looking for today? I'd be happy to show you our latest products."
            elif sentiment['category'] == 'positive':
                return "Great! How can I help you with your shopping today? Would you like to see our featured products or search for something specific?"
        
        # Default fallback response
        return "I'm not sure I understand what you're looking for. You can search for products, ask for recommendations, or say 'help' to see what I can do for you. What would you like to do?"
"""

# utils.py
utils_py = """
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for the e-commerce sentiment analysis system.
"""

import random
import datetime

def score_to_category(score):
    """Convert a sentiment score to a category.
    
    Args:
        score (float): Sentiment score between -1 and 1
        
    Returns:
        str: Sentiment category
    """
    if score >= 0.5:
        return "very positive"
    elif score >= 0.05:
        return "positive"
    elif score <= -0.5:
        return "very negative"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

def generate_farewell(sentiment_history):
    """Generate a farewell message based on sentiment history.
    
    Args:
        sentiment_history (list): List of sentiment analysis results
        
    Returns:
        str: Farewell message
    """
    if not sentiment_history:
        return "Thank you for using our service. Have a great day!"
    
    # Calculate average sentiment
    avg_sentiment = sum(s['compound'] for s in sentiment_history) / len(sentiment_history)
    
    # Generate sentiment-appropriate farewell
    if avg_sentiment >= 0.5:
        return "Thank you so much for your visit today! It's been a pleasure assisting you. Have a wonderful day!"
    elif avg_sentiment >= 0.05:
        return "Thanks for shopping with us today. I hope you found what you were looking for. Have a great day!"
    elif avg_sentiment <= -0.5:
        return "I apologize for any inconvenience during your visit. We value your feedback and will work to improve. Thank you for your patience."
    elif avg_sentiment <= -0.05:
        return "Thank you for your visit. If there's anything we could improve, please let us know. Have a good day."
    else:
        return "Thank you for using our service. Feel free to return if you need any assistance. Have a nice day!"

def enhance_response_with_sentiment(response, sentiment):
    """Enhance a response with sentiment-aware elements.
    
    Args:
        response (str): Original response text
        sentiment (dict): Sentiment analysis results
        
    Returns:
        str: Enhanced response
    """
    if not sentiment:
        return response
    
    sentiment_score = sentiment['compound']
    
    # For very negative sentiment, add empathy
    if sentiment_score <= -0.5:
        empathy_phrases = [
            "I understand your frustration. ",
            "I'm sorry to hear that. ",
            "I apologize for the inconvenience. "
        ]
        return random.choice(empathy_phrases) + response
    
    # For negative sentiment, add reassurance
    elif sentiment_score <= -0.05:
        reassurance_phrases = [
            "I'd like to help with that. ",
            "Let me try to address your concerns. ",
            "I'll do my best to assist you. "
        ]
        return random.choice(reassurance_phrases) + response
    
    # For very positive sentiment, add enthusiasm
    elif sentiment_score >= 0.5:
        enthusiasm_phrases = [
            "Wonderful! ",
            "Excellent! ",
            "That's great to hear! "
        ]
        return random.choice(enthusiasm_phrases) + response
    
    # For positive sentiment, add positivity
    elif sentiment_score >= 0.05:
        positive_phrases = [
            "Glad to hear that! ",
            "Happy to help! ",
            "Great! "
        ]
        return random.choice(positive_phrases) + response
    
    # For neutral sentiment, return original response
    return response

def get_time_appropriate_greeting():
    """Get a greeting appropriate for the current time of day.
    
    Returns:
        str: Time-appropriate greeting
    """
    current_hour = datetime.datetime.now().hour
    
    if 5 <= current_hour < 12:
        return "Good morning!"
    elif 12 <= current_hour < 18:
        return "Good afternoon!"
    else:
        return "Good evening!"
"""

# README file
readme_md = """
# Voice-Enabled E-commerce Sentiment Analysis System

## Project Overview

This project implements a voice-enabled e-commerce system that analyzes customer sentiment through spoken interactions. The system can:

1. Accept voice input from users
2. Convert speech to text
3. Analyze sentiment of user input
4. Respond appropriately based on detected emotions
5. Handle e-commerce functionality (product search, cart management, etc.)
6. Provide voice responses

## Installation Guide

### Prerequisites

- Python 3.7 or higher
- A microphone for voice input (optional, can run in text-only mode)
- Speakers for voice output (optional, can run in text-only mode)

### Setup Instructions

1. **Clone the repository or extract the project files to a directory**

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Note: PyAudio installation may require additional steps depending on your platform:
   - On Windows: `pip install PyAudio` should work directly
   - On macOS: `brew install portaudio` then `pip install PyAudio`
   - On Linux: `sudo apt-get install python3-pyaudio` or `sudo apt-get install portaudio19-dev` then `pip install PyAudio`

4. **Download NLTK resources**
   ```python
   python -c "import nltk; nltk.download('vader_lexicon')"
   ```

## Running the Application

### Basic Usage

To start the application with voice input/output:
```bash
python main.py
```

To run in text-only mode (no microphone/speakers needed):
```bash
python main.py --no-voice
```

To enable debug information (including sentiment scores):
```bash
python main.py --debug
```

### Interactive Commands

Once the application is running, you can interact with it using voice or text. Some example commands:

- **Search for products**: "Find smartphones" or "Show me kitchen appliances"
- **Get product details**: "Tell me about the Wireless Headphones"
- **Check prices**: "How much is the Coffee Maker?"
- **Add to cart**: "Add Smartwatch to my cart"
- **View cart**: "Show my cart"
- **Remove from cart**: "Remove Denim Jeans from my cart"
- **Checkout**: "Proceed to checkout"
- **Get help**: "What can you do?"
- **Exit**: "Goodbye" or "Exit"

## System Components

The system consists of several key modules:

1. **Voice Handler (`voice_handler.py`)**
   - Manages speech-to-text and text-to-speech operations
   - Handles microphone input and speaker output
   - Falls back to text input/output when voice is unavailable

2. **Sentiment Analyzer (`sentiment_analyzer.py`)**
   - Analyzes customer sentiment using VADER and/or Transformer models
   - Provides sentiment scores and categories (very negative, negative, neutral, positive, very positive)
   - Detects specific emotions where possible

3. **E-commerce System (`ecommerce_system.py`)**
   - Manages product catalog and shopping cart
   - Processes customer queries using intent recognition
   - Handles product searches, details, and recommendations

4. **Utilities (`utils.py`)**
   - Helper functions for sentiment processing
   - Response enhancement based on detected sentiment
   - Miscellaneous utility functions

5. **Main Application (`main.py`)**
   - Orchestrates all components
   - Manages the main interaction loop
   - Processes command-line arguments

## Customization

- **Product Catalog**: The system initializes with a demo product catalog. To use your own product data, create a `product_catalog.json` file following the same structure as the demo data.

- **Voice Settings**: You can adjust voice settings (rate, voice selection) in the `VoiceHandler` class.

- **Sentiment Thresholds**: Sentiment classification thresholds can be modified in the `score_to_category` function in `utils.py`.

## Troubleshooting

- **Microphone Issues**: If the system fails to initialize the microphone, it will automatically fall back to text-only mode. You can also force text-only mode with the `--no-voice` flag.

- **PyAudio Installation**: If you encounter issues installing PyAudio, refer to the platform-specific instructions in the Setup section.

- **Speech Recognition Errors**: If the system has trouble understanding speech, try speaking more clearly and reducing background noise. The `--debug` flag can help identify recognition issues.

## Project Showcase Tips

When showcasing this project:

1. **Prepare Demo Scenarios**: Have a few example customer journeys ready to demonstrate (e.g., searching for a product, adding to cart, checking out).

2. **Highlight Sentiment Adaptation**: Show how the system responds differently based on positive vs. negative sentiment.

3. **Demonstrate Voice and Text Modes**: Show both voice interaction and text-only modes to demonstrate flexibility.

4. **Use the Debug Flag**: Run with `--debug` to show the sentiment scores during your presentation.

5. **Explain Architecture**: Highlight the modular design and how the components interact.

## Future Enhancements

Potential areas for improvement:

1. Integration with actual e-commerce platforms
2. More sophisticated emotion detection
3. Customer profile and personalization
4. Multi-language support
5. Enhanced voice interaction with context awareness
"""

# Create a file with example usage
showcase_py = """
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Demo script to showcase the voice-enabled sentiment analysis e-commerce system.
This script runs through several predefined scenarios to demonstrate the system's capabilities.
"""

import time
from voice_handler import VoiceHandler
from sentiment_analyzer import SentimentAnalyzer
from ecommerce_system import EcommerceSystem
import utils

def run_demo():
    """Run through a series of predefined scenarios to showcase the system."""
    print("=" * 80)
    print("VOICE-ENABLED E-COMMERCE SENTIMENT ANALYSIS SYSTEM - DEMONSTRATION")
    print("=" * 80)
    
    # Initialize components
    print("\nInitializing system components...")
    voice_handler = VoiceHandler(text_only=True)  # Use text-only mode for the demo
    sentiment_analyzer = SentimentAnalyzer()
    ecommerce = EcommerceSystem()
    
    # Define demo scenarios
    scenarios = [
        {
            "name": "Positive product search",
            "input": "I'm looking for a new smartphone",
            "description": "Customer searching for a product with positive sentiment"
        },
        {
            "name": "Product information",
            "input": "Tell me more about the Smartphone XYZ",
            "description": "Customer asking for detailed product information"
        },
        {
            "name": "Add to cart",
            "input": "That sounds great, I'll add it to my cart",
            "description": "Customer adding product to cart with positive sentiment"
        },
        {
            "name": "Negative experience",
            "input": "This is taking too long and I'm getting frustrated",
            "description": "Customer expressing frustration (negative sentiment)"
        },
        {
            "name": "View cart",
            "input": "What's in my shopping cart?",
            "description": "Customer checking their shopping cart"
        },
        {
            "name": "Checkout",
            "input": "I want to check out now",
            "description": "Customer proceeding to checkout"
        },
        {
            "name": "Very positive feedback",
            "input": "This was an amazing shopping experience! Thank you so much for your help!",
            "description": "Customer providing very positive feedback"
        }
    ]
    
    # Run through scenarios
    for i, scenario in enumerate(scenarios):
        print("\n" + "-" * 80)
        print(f"DEMO SCENARIO {i+1}: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print("-" * 80)
        
        # Display user input
        user_input = scenario['input']
        print(f"USER: {user_input}")
        
        # Analyze sentiment
        sentiment = sentiment_analyzer.analyze(user_input)
        print(f"SENTIMENT ANALYSIS: {sentiment['category']} (Score: {sentiment['compound']:.2f})")
        
        # Process the query
        response = ecommerce.process_query(user_input, sentiment)
        
        # Enhance response with sentiment
        response = utils.enhance_response_with_sentiment(response, sentiment)
        
        # Display system response
        print(f"SYSTEM: {response}")
        
        # Add a pause between scenarios
        time.sleep(1)
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nTo run the full interactive system, execute: python main.py")
    print("For text-only mode: python main.py --no-voice")
    print("For debug information: python main.py --debug")

if __name__ == "__main__":
    run_demo()
"""

## Installation and Usage Guide

installation_guide = """
# Detailed Installation and Usage Guide

## Installation Steps

### 1. Environment Setup

First, create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\\Scripts\\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

#### Troubleshooting PyAudio Installation

PyAudio can be tricky to install on some systems:

**Windows:**
```bash
pip install PyAudio
```

**macOS:**
```bash
brew install portaudio
pip install PyAudio
```

**Linux:**
```bash
sudo apt-get install python3-pyaudio
# OR
sudo apt-get install portaudio19-dev
pip install PyAudio
```

### 3. Download NLTK Resources

```bash
python -c "import nltk; nltk.download('vader_lexicon')"
```

### 4. Download Pre-trained Models (Optional)

For better sentiment analysis performance, the system can use pre-trained transformer models from Hugging Face. The first time you run the application, it will automatically download the required models.

## Running the Application

### Basic Usage

```bash
# Run with voice input/output (default)
python main.py

# Run in text-only mode
python main.py --no-voice

# Run with debug information
python main.py --debug
```

### Interactive Demo

To run a quick demonstration of the system capabilities:

```bash
python showcase.py
```

### Example Interactions

Here are some example interactions you can try:

1. **Searching for products:**
   - "Do you have any smartphones?"
   - "Show me kitchen appliances"
   - "I'm looking for books"

2. **Product information:**
   - "Tell me about the Wireless Headphones"
   - "What are the features of the Smartwatch Pro?"
   - "Describe the Coffee Maker"

3. **Price inquiries:**
   - "How much is the Smartphone XYZ?"
   - "What's the price of the Denim Jeans?"
   - "How expensive is the Non-stick Cookware Set?"

4. **Shopping cart operations:**
   - "Add Wireless Headphones to my cart"
   - "Remove Coffee Maker from my cart"
   - "What's in my cart?"
   - "Checkout now"

5. **General assistance:**
   - "What can you do?"
   - "Help me find a gift"
   - "I need recommendations"

6. **Exit the application:**
   - "Goodbye"
   - "Exit"
   - "Quit"

## Presentation Tips

When showcasing this project:

### 1. Technical Setup

- Ensure your microphone and speakers are properly configured
- Test in a quiet environment for better speech recognition
- Have a backup plan to use text-only mode if needed

### 2. Demonstration Flow

A good demonstration flow might be:

1. Start by explaining the project's purpose and components
2. Show the system initialization
3. Demonstrate searching for products
4. Get details about a specific product
5. Add the product to the cart
6. Show how the system responds to different sentiment:
   - Use a positive tone: "This looks fantastic!"
   - Use a negative tone: "This is too expensive and frustrating"
7. Complete the purchase
8. End with positive feedback to show sentiment adaptation

### 3. Key Points to Highlight

- The sentiment analysis component and how it adapts responses
- The voice recognition and synthesis capabilities
- The natural language understanding for e-commerce intents
- The modular architecture of the system
"""

# Return complete project
return """
# DEVELOPMENT OF A MODEL FOR CUSTOMER SENTIMENT ANALYSIS IN VOICE-ENABLED E-COMMERCE

This project implements a voice-enabled e-commerce system that can analyze customer sentiment
through spoken interactions. The system transcribes voice input, analyzes the sentiment,
and responds appropriately based on detected customer emotions.

## Project Structure

- `main.py`: Core application entry point
- `voice_handler.py`: Voice input/output functionality
- `sentiment_analyzer.py`: Sentiment analysis model
- `ecommerce_system.py`: Mock e-commerce system
- `utils.py`: Helper functions and utilities
- `requirements.txt`: Required dependencies

## Setup and Installation Guide

# Let's start with requirements.txt file content
import textwrap

requirements = """
# Core libraries
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2

# Voice processing
SpeechRecognition==3.10.0
PyAudio==0.2.13
pyttsx3==2.90
gTTS==2.3.2

# NLP and Sentiment Analysis
nltk==3.8.1
transformers==4.30.2
torch==2.0.1
scikit-learn==1.3.0

# Utilities
tqdm==4.65.0
python-dotenv==1.0.0
"""

# main.py - Core application
main_py = """
"""

main_py = """
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DEVELOPMENT OF A MODEL FOR CUSTOMER SENTIMENT ANALYSIS IN VOICE-ENABLED E-COMMERCE

This is the main application entry point that brings together all components:
- Voice input/output handling
- Sentiment analysis
- E-commerce system interaction
"""

import os
import time
import argparse
from voice_handler import VoiceHandler
from sentiment_analyzer import SentimentAnalyzer
from ecommerce_system import EcommerceSystem
import utils

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Voice-Enabled E-commerce Sentiment Analysis')
    parser.add_argument('--no-voice', action='store_true', help='Run in text-only mode (no voice)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    return parser.parse_args()

def main():
    """Main application entry point."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize components
    print("Initializing system components...")
    voice_handler = VoiceHandler(text_only=args.no_voice)
    sentiment_analyzer = SentimentAnalyzer()
    ecommerce = EcommerceSystem()
    
    # Application state
    session_active = True
    customer_name = None
    shopping_cart = []
    sentiment_history = []
    
    # Welcome message
    welcome_message = "Welcome to our voice-enabled e-commerce system! How can I help you today?"
    print("\n" + "-"*80)
    print("SYSTEM: " + welcome_message)
    voice_handler.speak(welcome_message)
    
    # Main interaction loop
    while session_active:
        # Get user input
        print("\n" + "-"*40)
        user_input = voice_handler.listen()
        if not user_input:
            continue
            
        print(f"YOU: {user_input}")
        
        # Process commands
        if "exit" in user_input.lower() or "quit" in user_input.lower() or "goodbye" in user_input.lower():
            farewell = utils.generate_farewell(sentiment_history)
            print("SYSTEM: " + farewell)
            voice_handler.speak(farewell)
            session_active = False
            continue
        
        # Analyze sentiment
        sentiment = sentiment_analyzer.analyze(user_input)
        sentiment_score = sentiment['compound']
        sentiment_category = sentiment['category']
        sentiment_history.append(sentiment)
        
        if args.debug:
            print(f"DEBUG - Sentiment: {sentiment_category} (Score: {sentiment_score:.2f})")
        
        # Process e-commerce interactions
        response = ecommerce.process_query(user_input, sentiment)
        
        # Add sentiment-aware elements to response
        response = utils.enhance_response_with_sentiment(response, sentiment)
        
        # Output response
        print("SYSTEM: " + response)
        voice_handler.speak(response)
    
    # Session summary
    if len(sentiment_history) > 1:
        avg_sentiment = sum(s['compound'] for s in sentiment_history) / len(sentiment_history)
        print("\n" + "="*80)
        print(f"Session Summary:")
        print(f"Total interactions: {len(sentiment_history)}")
        print(f"Average sentiment: {avg_sentiment:.2f} ({utils.score_to_category(avg_sentiment)})")
        print("="*80)
    
    print("\nThank you for using our voice-enabled e-commerce system!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram terminated by user.")
    except Exception as e:
        print(f"\n\nAn error occurred: {e}")
"""

# voice_handler.py
voice_handler_py = """
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Voice input/output handling component for the e-commerce sentiment analysis system.
This module provides speech-to-text and text-to-speech functionality.
"""

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
            print ${total:.2f}."
            
            # Add new item to cart
            self.shopping_cart.append({
                'product': product,
                'quantity': 1
            })
            
            total = sum(item['product']['price'] * item['quantity'] for item in self.shopping_cart)
            return f"Added {product['name']} to your cart. Your cart total is now ${total:.2f}. Would you like to continue shopping or view your cart?"
        else:
            return f"I couldn't find '{product_name}' in our catalog. Could you try again with a different product name?"
    
    def _handle_remove_from_cart(self, product_name):
        """Handle removing products from cart.
        
        Args:
            product_name (str): Name of product to remove
            
        Returns:
            str: Confirmation message
        """
        if not self.shopping_cart:
            return "Your shopping cart is empty."
        
        product = self._find_product(product_name)
        
        if not product:
            return f"I couldn't find '{product_name}' in our catalog or your cart."
        
        for i, item in enumerate(self.shopping_cart):
            if item['product']['id'] == product['id']:
                if item['quantity'] > 1:
                    item['quantity'] -= 1
                    total = sum(item['product']['price'] * item['quantity'] for item in self.shopping_cart)
                    return f"Removed one {product['name']} from your cart. You now have {item['quantity']} in your cart. Your cart total is