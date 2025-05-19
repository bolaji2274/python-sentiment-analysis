# main_py = """
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
