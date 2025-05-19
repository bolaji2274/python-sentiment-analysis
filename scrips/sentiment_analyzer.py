
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