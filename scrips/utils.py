# utils_py = """
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# Utility functions for the e-commerce sentiment analysis system.
# """

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