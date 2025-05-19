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
