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
        self.wishlist = []  # New feature: wishlist for saving products for later
        self.user_preferences = {  # New feature: track user preferences
            'favorite_categories': [],
            'price_range': {'min': 0, 'max': float('inf')},
            'brands': []
        }
        
        # Intent patterns
        # self.intent_patterns = {
        #     'search': r'(?:find|search|available|looking for|show me|do you have|searching for|need to find)\s+(.+)',
        #     'info': r'(?:tell me about|details on|more information about|specs for|features of|describe)\s+(.+)',
        #     'price': r'(?:how much|price of|cost of|what is the price of|how expensive is)\s+(.+)',
        #     'add_to_cart': r'(?:add|put|place)\s+(.+?)\s+(?:to|in|into|on)(?:\s+my)?\s+(?:cart|basket)',
        #     'remove_from_cart': r'(?:remove|take out|delete)\s+(.+?)\s+(?:from)(?:\s+my)?\s+(?:cart|basket)',
        #     'view_cart': r'(?:view|show|what\'s in|display|see)(?:\s+my)?\s+(?:cart|basket)',
        #     'checkout': r'(?:checkout|proceed to checkout|buy now|purchase|complete purchase|place order)',
        #     'help': r'(?:help|assist|support|how do I|how to|what can you do)',
        #     'recommendations': r'(?:recommend|suggestion|what do you recommend|popular items)',
        #     'greeting': r'(?:hi|hello|hey|greetings)',
        #     'feedback': r'(?:feedback|review|rate|comment)'
        # }
        # Intent patterns - expanded with more shopping intents
        self.intent_patterns = {
            'search': r'(?:find|search|available|looking for|show me|do you have|searching for|need to find|browse|get|locate)\s+(.+)',
            'info': r'(?:tell me about|details on|more information about|specs for|features of|describe|what about|info on|specifications for|tell me more about)\s+(.+)',
            'price': r'(?:how much|price of|cost of|what is the price of|how expensive is|what does it cost for|pricing for|price info for|price details for)\s+(.+)',
            'add_to_cart': r'(?:add|put|place|include|insert)\s+(.+?)\s+(?:to|in|into|on)(?:\s+my)?\s+(?:cart|basket|shopping bag|order)',
            'remove_from_cart': r'(?:remove|take out|delete|get rid of|exclude)\s+(.+?)\s+(?:from)(?:\s+my)?\s+(?:cart|basket|shopping bag|order)',
            'view_cart': r'(?:view|show|what\'s in|display|see|check|review)(?:\s+my)?\s+(?:cart|basket|shopping bag|order)',
            'checkout': r'(?:checkout|proceed to checkout|buy now|purchase|complete purchase|place order|finalize order|pay|complete transaction)',
            'help': r'(?:help|assist|support|how do I|how to|what can you do|guide me|instructions|assistance)',
            'recommendations': r'(?:recommend|suggestion|what do you recommend|popular items|best sellers|trending items|what\'s popular|top rated|best products)',
            'greeting': r'(?:hi|hello|hey|greetings|good morning|good afternoon|good evening)',
            'feedback': r'(?:feedback|review|rate|comment|opinion|thoughts|critique)',
            # New intents
            'compare_products': r'(?:compare|difference between|versus|vs|which is better)(?:\s+between)?\s+(.+)',
            'availability': r'(?:is|are)(?:\s+there)?\s+(.+?)\s+(?:in stock|available|in inventory)',
            'add_to_wishlist': r'(?:add|save|put)\s+(.+?)\s+(?:to|in|into|on)(?:\s+my)?\s+(?:wishlist|favorites|saved items|wish list)',
            'view_wishlist': r'(?:view|show|what\'s in|display|see)(?:\s+my)?\s+(?:wishlist|favorites|saved items|wish list)',
            'sort_by': r'(?:sort|arrange|order|filter)(?:\s+products)?\s+by\s+(.+)',
            'sales': r'(?:sales|discounts|deals|promotions|special offers|clearance items|bargains|coupons)',
            'shipping': r'(?:shipping|delivery|shipping options|shipping cost|delivery time|when will it arrive|shipping policy)',
            'return_policy': r'(?:return policy|returns|refunds|can I return|how to return|refund policy)',
            'new_arrivals': r'(?:new arrivals|new products|latest items|just in|recently added|what\'s new)',
            'brand_search': r'(?:products from|items by|made by|brand)\s+(.+)'
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
        
        # Electronics - expanded
        products.extend([
            {
                'id': 'e1',
                'name': 'Smartphone XYZ',
                'category': 'Electronics',
                'subcategory': 'Smartphones',
                'brand': 'TechCorp',
                'price': 699.99,
                'description': 'Latest smartphone with 6.7-inch display, 128GB storage, and triple camera system.',
                'rating': 4.5,
                'stock': 25,
                'features': ['5G compatible', 'Water resistant', 'Face recognition', 'Wireless charging'],
                'colors': ['Black', 'Silver', 'Blue'],
                'warranty': '1 year'
            },
            {
                'id': 'e2',
                'name': 'Wireless Headphones',
                'category': 'Electronics',
                'subcategory': 'Audio',
                'brand': 'SoundMaster',
                'price': 149.99,
                'description': 'Premium noise-cancelling wireless headphones with 30-hour battery life.',
                'rating': 4.7,
                'stock': 40,
                'features': ['Active noise cancellation', 'Bluetooth 5.0', 'Voice assistant support'],
                'colors': ['Black', 'White', 'Rose Gold'],
                'warranty': '2 years'
            },
            {
                'id': 'e3',
                'name': 'Smartwatch Pro',
                'category': 'Electronics',
                'subcategory': 'Wearables',
                'brand': 'FitTech',
                'price': 249.99,
                'description': 'Advanced smartwatch with health monitoring, GPS, and waterproof design.',
                'rating': 4.3,
                'stock': 15,
                'features': ['Heart rate monitor', 'Sleep tracking', 'GPS', 'Water resistant 50m'],
                'colors': ['Black', 'Silver'],
                'warranty': '1 year'
            },
            {
                'id': 'e4',
                'name': 'Ultra HD 4K Smart TV',
                'category': 'Electronics',
                'subcategory': 'Televisions',
                'brand': 'ViewTech',
                'price': 899.99,
                'description': '55-inch Ultra HD smart TV with HDR and voice control.',
                'rating': 4.6,
                'stock': 10,
                'features': ['4K resolution', 'HDR support', 'Voice control', 'Smart TV apps'],
                'colors': ['Black'],
                'warranty': '2 years'
            },
            {
                'id': 'e5',
                'name': 'Gaming Laptop',
                'category': 'Electronics',
                'subcategory': 'Computers',
                'brand': 'GameEdge',
                'price': 1299.99,
                'description': 'High-performance gaming laptop with RTX graphics and 16GB RAM.',
                'rating': 4.8,
                'stock': 8,
                'features': ['RTX 3070 Graphics', '16GB RAM', '1TB SSD', '144Hz display'],
                'colors': ['Black', 'Gray'],
                'warranty': '1 year'
            },
            {
                'id': 'e6',
                'name': 'Wireless Earbuds',
                'category': 'Electronics',
                'subcategory': 'Audio',
                'brand': 'SoundMaster',
                'price': 129.99,
                'description': 'True wireless earbuds with active noise cancellation and water resistance.',
                'rating': 4.4,
                'stock': 30,
                'features': ['Active noise cancellation', 'Water resistant', '24-hour battery life'],
                'colors': ['Black', 'White', 'Blue'],
                'warranty': '1 year'
            },
            {
                'id': 'e7',
                'name': 'Digital Camera Pro',
                'category': 'Electronics',
                'subcategory': 'Cameras',
                'brand': 'PhotoMax',
                'price': 799.99,
                'description': 'Professional digital camera with 24MP sensor and 4K video recording.',
                'rating': 4.7,
                'stock': 12,
                'features': ['24MP sensor', '4K video', 'Image stabilization', 'Wi-Fi connectivity'],
                'colors': ['Black'],
                'warranty': '2 years'
            }
        ])
        
        # Clothing - expanded
        products.extend([
            {
                'id': 'c1',
                'name': 'Casual T-shirt',
                'category': 'Clothing',
                'subcategory': 'Tops',
                'brand': 'ComfortWear',
                'price': 19.99,
                'description': 'Comfortable cotton t-shirt available in multiple colors.',
                'rating': 4.2,
                'stock': 100,
                'features': ['100% cotton', 'Machine washable'],
                'sizes': ['S', 'M', 'L', 'XL'],
                'colors': ['White', 'Black', 'Blue', 'Red', 'Gray']
            },
            {
                'id': 'c2',
                'name': 'Denim Jeans',
                'category': 'Clothing',
                'subcategory': 'Bottoms',
                'brand': 'DenimCo',
                'price': 59.99,
                'description': 'Classic denim jeans with straight fit.',
                'rating': 4.4,
                'stock': 75,
                'features': ['Cotton blend', 'Straight fit', 'Machine washable'],
                'sizes': ['30x32', '32x32', '34x32', '36x32'],
                'colors': ['Blue', 'Black', 'Gray']
            },
            {
                'id': 'c3',
                'name': 'Winter Jacket',
                'category': 'Clothing',
                'subcategory': 'Outerwear',
                'brand': 'OutdoorLife',
                'price': 129.99,
                'description': 'Insulated winter jacket with waterproof exterior.',
                'rating': 4.6,
                'stock': 40,
                'features': ['Waterproof', 'Insulated', 'Hood included', 'Multiple pockets'],
                'sizes': ['S', 'M', 'L', 'XL', 'XXL'],
                'colors': ['Black', 'Navy', 'Green']
            },
            {
                'id': 'c4',
                'name': 'Athletic Shoes',
                'category': 'Clothing',
                'subcategory': 'Footwear',
                'brand': 'SportStep',
                'price': 89.99,
                'description': 'Lightweight athletic shoes with cushioned soles.',
                'rating': 4.5,
                'stock': 60,
                'features': ['Breathable mesh', 'Cushioned sole', 'Arch support'],
                'sizes': ['7', '8', '9', '10', '11', '12'],
                'colors': ['Black/White', 'Gray/Blue', 'Red/Black']
            },
            {
                'id': 'c5',
                'name': 'Summer Dress',
                'category': 'Clothing',
                'subcategory': 'Dresses',
                'brand': 'StyleFusion',
                'price': 45.99,
                'description': 'Light and flowing summer dress with floral pattern.',
                'rating': 4.3,
                'stock': 50,
                'features': ['Lightweight fabric', 'Floral pattern', 'V-neck'],
                'sizes': ['XS', 'S', 'M', 'L', 'XL'],
                'colors': ['Blue Floral', 'Pink Floral', 'Yellow Floral']
            }
        ])
        
        # Home & Kitchen - expanded
        products.extend([
            {
                'id': 'h1',
                'name': 'Coffee Maker',
                'category': 'Home & Kitchen',
                'subcategory': 'Small Appliances',
                'brand': 'BrewMaster',
                'price': 89.99,
                'description': 'Programmable coffee maker with 12-cup capacity.',
                'rating': 4.1,
                'stock': 30,
                'features': ['Programmable timer', '12-cup capacity', 'Auto shut-off'],
                'colors': ['Black', 'Silver'],
                'warranty': '1 year'
            },
            {
                'id': 'h2',
                'name': 'Non-stick Cookware Set',
                'category': 'Home & Kitchen',
                'subcategory': 'Cookware',
                'brand': 'KitchenElite',
                'price': 129.99,
                'description': '10-piece non-stick cookware set with glass lids.',
                'rating': 4.6,
                'stock': 20,
                'features': ['Non-stick coating', 'Dishwasher safe', 'Glass lids', 'Stay-cool handles'],
                'colors': ['Black', 'Copper'],
                'warranty': '3 years'
            },
            {
                'id': 'h3',
                'name': 'Robot Vacuum Cleaner',
                'category': 'Home & Kitchen',
                'subcategory': 'Appliances',
                'brand': 'CleanTech',
                'price': 299.99,
                'description': 'Smart robot vacuum with mapping technology and app control.',
                'rating': 4.7,
                'stock': 15,
                'features': ['Smart mapping', 'App control', 'Automatic charging', 'HEPA filter'],
                'colors': ['White', 'Black'],
                'warranty': '2 years'
            },
            {
                'id': 'h4',
                'name': 'Memory Foam Mattress',
                'category': 'Home & Kitchen',
                'subcategory': 'Bedroom',
                'brand': 'DreamSleep',
                'price': 599.99,
                'description': 'Queen-sized memory foam mattress with cooling gel technology.',
                'rating': 4.8,
                'stock': 10,
                'features': ['Memory foam', 'Cooling gel', 'Hypoallergenic cover'],
                'sizes': ['Twin', 'Full', 'Queen', 'King'],
                'warranty': '10 years'
            },
            {
                'id': 'h5',
                'name': 'Air Fryer',
                'category': 'Home & Kitchen',
                'subcategory': 'Small Appliances',
                'brand': 'CrispTech',
                'price': 119.99,
                'description': 'Digital air fryer with 5.5-quart capacity and multiple cooking modes.',
                'rating': 4.5,
                'stock': 25,
                'features': ['5.5-quart capacity', 'Digital controls', '8 cooking presets', 'Dishwasher safe parts'],
                'colors': ['Black', 'White'],
                'warranty': '1 year'
            },
            {
                'id': 'h6',
                'name': 'Stand Mixer',
                'category': 'Home & Kitchen',
                'subcategory': 'Small Appliances',
                'brand': 'BakePro',
                'price': 249.99,
                'description': 'Professional stand mixer with 5-quart bowl and multiple attachments.',
                'rating': 4.9,
                'stock': 18,
                'features': ['5-quart bowl', '10-speed settings', '3 attachments included'],
                'colors': ['Red', 'Black', 'Silver', 'Blue'],
                'warranty': '3 years'
            }
        ])
        
        # Books - expanded
        products.extend([
            {
                'id': 'b1',
                'name': 'Artificial Intelligence Basics',
                'category': 'Books',
                'subcategory': 'Technology',
                'author': 'Dr. Alan Smith',
                'price': 29.99,
                'description': 'Introduction to artificial intelligence and machine learning concepts.',
                'rating': 4.8,
                'stock': 50,
                'features': ['Hardcover', '350 pages', 'Published 2023'],
                'format': ['Hardcover', 'Paperback', 'eBook']
            },
            {
                'id': 'b2',
                'name': 'The Bestseller Novel',
                'category': 'Books',
                'subcategory': 'Fiction',
                'author': 'Emma Johnson',
                'price': 14.99,
                'description': 'Award-winning fiction novel with over 1 million copies sold.',
                'rating': 4.9,
                'stock': 60,
                'features': ['Paperback', '400 pages', 'Published 2023'],
                'format': ['Hardcover', 'Paperback', 'eBook', 'Audiobook']
            },
            {
                'id': 'b3',
                'name': 'Cooking Around the World',
                'category': 'Books',
                'subcategory': 'Cookbooks',
                'author': 'Chef Marco Torres',
                'price': 34.99,
                'description': 'Collection of international recipes from 50 different countries.',
                'rating': 4.7,
                'stock': 30,
                'features': ['Hardcover', '250 recipes', 'Color photos', 'Published 2023'],
                'format': ['Hardcover', 'eBook']
            },
            {
                'id': 'b4',
                'name': 'Financial Freedom Guide',
                'category': 'Books',
                'subcategory': 'Finance',
                'author': 'Sarah Miller',
                'price': 19.99,
                'description': 'Step-by-step guide to achieving financial independence and early retirement.',
                'rating': 4.6,
                'stock': 40,
                'features': ['Paperback', '320 pages', 'Published 2022'],
                'format': ['Paperback', 'eBook', 'Audiobook']
            },
            {
                'id': 'b5',
                'name': 'Mystery at Midnight',
                'category': 'Books',
                'subcategory': 'Mystery',
                'author': 'Thomas Black',
                'price': 12.99,
                'description': 'Thrilling mystery novel featuring detective Alex Grey.',
                'rating': 4.5,
                'stock': 35,
                'features': ['Paperback', '280 pages', 'Published 2023'],
                'format': ['Hardcover', 'Paperback', 'eBook', 'Audiobook']
            }
        ])
        
        # Beauty - expanded
        products.extend([
            {
                'id': 'be1',
                'name': 'Face Serum',
                'category': 'Beauty',
                'subcategory': 'Skincare',
                'brand': 'GlowUp',
                'price': 24.99,
                'description': 'Hydrating face serum with vitamin C for all skin types.',
                'rating': 4.5,
                'stock': 45,
                'features': ['Vitamin C', 'Hyaluronic acid', 'Paraben-free', '30ml'],
                'skin_type': ['All skin types']
            },
            {
                'id': 'be2',
                'name': 'Makeup Set',
                'category': 'Beauty',
                'subcategory': 'Makeup',
                'brand': 'BeautyGlam',
                'price': 39.99,
                'description': 'Complete makeup set with eyeshadow, lipstick, and mascara.',
                'rating': 4.3,
                'stock': 35,
                'features': ['12-color eyeshadow palette', '2 lipsticks', 'Volumizing mascara'],
                'colors': ['Neutral', 'Bold']
            },
            {
                'id': 'be3',
                'name': 'Moisturizing Shampoo',
                'category': 'Beauty',
                'subcategory': 'Hair Care',
                'brand': 'HairLux',
                'price': 14.99,
                'description': 'Moisturizing shampoo for dry and damaged hair.',
                'rating': 4.4,
                'stock': 50,
                'features': ['Argan oil', 'Sulfate-free', '16 fl oz'],
                'hair_type': ['Dry', 'Damaged']
            },
            {
                'id': 'be4',
                'name': 'Perfume Collection',
                'category': 'Beauty',
                'subcategory': 'Fragrance',
                'brand': 'ScentSation',
                'price': 79.99,
                'description': 'Collection of three premium fragrances in travel sizes.',
                'rating': 4.7,
                'stock': 20,
                'features': ['3 x 15ml bottles', 'Long-lasting', 'Gift box included'],
                'scents': ['Floral', 'Citrus', 'Woody']
            },
            {
                'id': 'be5',
                'name': 'Anti-Aging Cream',
                'category': 'Beauty',
                'subcategory': 'Skincare',
                'brand': 'AgelessBeauty',
                'price': 49.99,
                'description': 'Premium anti-aging face cream with retinol and peptides.',
                'rating': 4.8,
                'stock': 30,
                'features': ['Retinol', 'Peptides', 'Hyaluronic acid', '50ml jar'],
                'skin_type': ['Mature skin', 'All skin types']
            }
        ])
        
        # New category: Sports & Outdoors
        products.extend([
            {
                'id': 's1',
                'name': 'Yoga Mat',
                'category': 'Sports & Outdoors',
                'subcategory': 'Yoga',
                'brand': 'ZenFlex',
                'price': 29.99,
                'description': 'Non-slip yoga mat with carrying strap, 6mm thickness.',
                'rating': 4.6,
                'stock': 40,
                'features': ['6mm thickness', 'Non-slip surface', 'Carrying strap included'],
                'colors': ['Purple', 'Blue', 'Black', 'Green']
            },
            {
                'id': 's2',
                'name': 'Mountain Bike',
                'category': 'Sports & Outdoors',
                'subcategory': 'Cycling',
                'brand': 'TrailBlazer',
                'price': 549.99,
                'description': '27-speed mountain bike with front suspension and disc brakes.',
                'rating': 4.7,
                'stock': 10,
                'features': ['27-speed', 'Front suspension', 'Hydraulic disc brakes', 'Aluminum frame'],
                'colors': ['Black/Red', 'Blue/Silver'],
                'sizes': ['16"', '18"', '20"'],
                'warranty': '2 years'
            },
            {
                'id': 's3',
                'name': 'Tennis Racket',
                'category': 'Sports & Outdoors',
                'subcategory': 'Tennis',
                'brand': 'AcePro',
                'price': 119.99,
                'description': 'Professional tennis racket with carbon fiber frame.',
                'rating': 4.5,
                'stock': 25,
                'features': ['Carbon fiber frame', 'Oversized head', 'Vibration dampening'],
                'grip_size': ['4 1/8"', '4 1/4"', '4 3/8"'],
                'warranty': '1 year'
            },
            {
                'id': 's4',
                'name': 'Camping Tent',
                'category': 'Sports & Outdoors',
                'subcategory': 'Camping',
                'brand': 'OutdoorLife',
                'price': 149.99,
                'description': '4-person waterproof tent with quick setup system.',
                'rating': 4.6,
                'stock': 15,
                'features': ['4-person capacity', 'Waterproof', 'UV protection', 'Mesh windows'],
                'colors': ['Green', 'Orange', 'Blue'],
                'warranty': '1 year'
            }
        ])
        
        # New category: Toys & Games
        products.extend([
            {
                'id': 't1',
                'name': 'Building Blocks Set',
                'category': 'Toys & Games',
                'subcategory': 'Construction Toys',
                'brand': 'BlockMaster',
                'price': 34.99,
                'description': '250-piece colorful building blocks set compatible with major brands.',
                'rating': 4.8,
                'stock': 30,
                'features': ['250 pieces', 'Compatible with major brands', 'Storage container included'],
                'age_range': '4-12 years'
            },
            {
                'id': 't2',
                'name': 'Board Game Collection',
                'category': 'Toys & Games',
                'subcategory': 'Board Games',
                'brand': 'GameNight',
                'price': 49.99,
                'description': 'Collection of 5 classic board games in one package.',
                'rating': 4.7,
                'stock': 20,
                'features': ['5 games included', 'Family friendly', '2-6 players'],
                'age_range': '6+ years'
            },
            {
                'id': 't3',
                'name': 'Remote Control Car',
                'category': 'Toys & Games',
                'subcategory': 'Remote Control',
                'brand': 'SpeedRacer',
                'price': 69.99,
                'description': 'High-speed remote control car with rechargeable battery.',
                'rating': 4.5,
                'stock': 15,
                'features': ['30 mph top speed', 'Rechargeable battery', '100m control range'],
                'colors': ['Red', 'Blue', 'Green'],
                'age_range': '8+ years',
                'warranty': '6 months'
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
                # New intent handlers
                elif intent == 'compare_products':
                    return self._handle_compare_products(entity)
                elif intent == 'availability':
                    return self._handle_availability(entity)
                elif intent == 'add_to_wishlist':
                    return self._handle_add_to_wishlist(entity)
                elif intent == 'view_wishlist':
                    return self._handle_view_wishlist()
                elif intent == 'sort_by':
                    return self._handle_sort_by(entity)
                elif intent == 'sales':
                    return self._handle_sales()
                elif intent == 'shipping':
                    return self._handle_shipping()
                elif intent == 'return_policy':
                    return self._handle_return_policy()
                elif intent == 'new_arrivals':
                    return self._handle_new_arrivals()
                elif intent == 'brand_search':
                    return self._handle_brand_search(entity)
        
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
    
    def _find_products_by_category(self, category):
        """Find all products in a specific category.
        
        Args:
            category (str): Category to search for
            
        Returns:
            list: List of products in the category
        """
        if not category:
            return []
            
        category = category.lower()
        
        # Try exact category match
        exact_matches = [p for p in self.products if p['category'].lower() == category]
        if exact_matches:
            return exact_matches
            
        # Try subcategory match
        subcategory_matches = [p for p in self.products if 'subcategory' in p and p['subcategory'].lower() == category]
        if subcategory_matches:
            return subcategory_matches
            
        # Try partial category match
        partial_matches = [p for p in self.products if category in p['category'].lower()]
        if partial_matches:
            return partial_matches
            
        # Try partial subcategory match
        partial_subcategory_matches = [p for p in self.products if 'subcategory' in p and category in p['subcategory'].lower()]
        if partial_subcategory_matches:
            return partial_subcategory_matches
            
        return []
    def _handle_compare_products(self, products_str):
        """Handle product comparison requests.
        
        Args:
            products_str (str): String containing products to compare
            
        Returns:
            str: Product comparison
        """
        if not products_str:
            return "Please specify which products you'd like to compare."
        
        # Try to identify two products in the query
        product_names = []
        
        # Check for "and" or "vs" or "versus" or "or" to split the products
        for separator in [' and ', ' vs ', ' versus ', ' or ']:
            if separator in products_str.lower():
                product_names = [name.strip() for name in products_str.split(separator)]
                break
        
        # If no separator found, try to find the products in our catalog
        if not product_names:
            # Check each product name in our catalog
            for product in self.products:
                if product['name'].lower() in products_str.lower():
                    product_names.append(product['name'])
                    
            # If still didn't find two products, return error
            if len(product_names) < 2:
                return "I couldn't identify two distinct products to compare. Please specify them clearly, for example: 'Compare Smartphone XYZ and Wireless Headphones'."
        
        # Find the product objects
        products_to_compare = []
        for name in product_names[:2]:  # Limit to comparing 2 products
            product = self._find_product(name)
            if product:
                products_to_compare.append(product)
                
        if len(products_to_compare) < 2:
            return "I couldn't find both products in our catalog. Please check the product names and try again."
        
        # Build comparison
        p1 = products_to_compare[0]
        p2 = products_to_compare[1]
        
        response = f"Comparing {p1['name']} and {p2['name']}:\n\n"
        
        # Price comparison
        response += f"Price:\n- {p1['name']}: ${p1['price']:.2f}\n- {p2['name']}: ${p2['price']:.2f}\n"
        price_diff = abs(p1['price'] - p2['price'])
        cheaper = p1['name'] if p1['price'] < p2['price'] else p2['name']
        response += f"  • {cheaper} is ${price_diff:.2f} cheaper\n\n"
        
        # Rating comparison
        response += f"Rating:\n- {p1['name']}: {p1['rating']}/5.0\n- {p2['name']}: {p2['rating']}/5.0\n"
        better_rated = p1['name'] if p1['rating'] > p2['rating'] else p2['name']
        rating_diff = abs(p1['rating'] - p2['rating'])
        response += f"  • {better_rated} is rated {rating_diff:.1f} points higher\n\n"
        
        # Features comparison (if available)
        if 'features' in p1 and 'features' in p2:
            response += "Key Features:\n"
            response += f"- {p1['name']}: {', '.join(p1['features'][:3])}\n"
            response += f"- {p2['name']}: {', '.join(p2['features'][:3])}\n\n"
        
        # Stock comparison
        response += f"Availability:\n- {p1['name']}: {p1['stock']} in stock\n- {p2['name']}: {p2['stock']} in stock\n\n"
        
        # Recommendation
        if p1['rating'] > p2['rating'] and p1['price'] <= p2['price']:
            recommended = p1['name']
            reason = "higher rating at a similar or lower price"
        elif p2['rating'] > p1['rating'] and p2['price'] <= p1['price']:
            recommended = p2['name']
            reason = "higher rating at a similar or lower price"
        elif p1['rating'] == p2['rating']:
            recommended = cheaper
            reason = "same rating but more affordable"
        else:
            # More expensive product has better rating, compare value
            p1_value = p1['rating'] / p1['price']
            p2_value = p2['rating'] / p2['price']
            if p1_value > p2_value:
                recommended = p1['name']
                reason = "better value for money based on rating per dollar"
            else:
                recommended = p2['name']
                reason = "better value for money based on rating per dollar"
        
        response += f"Recommendation: The {recommended} offers {reason}.\n\n"
        response += "Would you like to add one of these products to your cart?"
        
        return response
    
    def _handle_availability(self, product_name):
        """Handle product availability inquiries.
        
        Args:
            product_name (str): Name of product to check availability
            
        Returns:
            str: Availability information
        """
        product = self._find_product(product_name)
        
        if not product:
            return f"I couldn't find a product named '{product_name}' in our catalog."
        
        if product['stock'] > 20:
            status = "plenty of stock"
        elif product['stock'] > 10:
            status = "good availability"
        elif product['stock'] > 5:
            status = "limited stock"
        elif product['stock'] > 0:
            status = "very low stock, almost sold out"
        else:
            status = "currently out of stock"
        
        if product['stock'] > 0:
            return f"Yes, the {product['name']} is in stock! We currently have {product['stock']} units available ({status}). Would you like to add it to your cart?"
        else:
            # Recommend similar products
            category_products = [p for p in self.products if p['category'] == product['category'] and p['id'] != product['id'] and p['stock'] > 0]
            if category_products:
                similar = sorted(category_products, key=lambda x: x['rating'], reverse=True)[0]
                return f"I'm sorry, the {product['name']} is {status}. Would you like me to notify you when it's back in stock? In the meantime, you might be interested in the {similar['name']} which is similar and currently available."
            else:
                return f"I'm sorry, the {product['name']} is {status}. Would you like me to notify you when it's back in stock?"
    
    def _handle_add_to_wishlist(self, product_name):
        """Handle adding products to wishlist.
        
        Args:
            product_name (str): Name of product to add to wishlist
            
        Returns:
            str: Confirmation message
        """
        product = self._find_product(product_name)
        
        if not product:
            return f"I couldn't find a product named '{product_name}' in our catalog."
        
        # Check if already in wishlist
        for item in self.wishlist:
            if item['id'] == product['id']:
                return f"The {product['name']} is already in your wishlist. Would you like to add it to your cart instead?"
        
        # Add to wishlist
        self.wishlist.append(product)
        
        return f"Added {product['name']} to your wishlist. You now have {len(self.wishlist)} items in your wishlist. Would you like to continue shopping?"
    
    def _handle_view_wishlist(self):
        """Handle requests to view wishlist.
        
        Returns:
            str: Wishlist contents
        """
        if not self.wishlist:
            return "Your wishlist is currently empty. As you browse our products, you can add items to your wishlist to save them for later."
        
        if len(self.wishlist) == 1:
            item = self.wishlist[0]
            return f"You have 1 item in your wishlist: {item['name']} (${item['price']:.2f}). Would you like to add it to your cart or remove it from your wishlist?"
        else:
            items = ", ".join([f"{p['name']} (${p['price']:.2f})" for p in self.wishlist[:5]])
            if len(self.wishlist) > 5:
                items += f", and {len(self.wishlist) - 5} more"
            
            return f"Your wishlist contains {len(self.wishlist)} items: {items}. Would you like to add any of these to your cart?"
    
    # def _handle_sort_by(self, sort_criterion):
    def _handle_sort_by(self, sort_criterion):
        """Handle sorting products by different criteria.
        
        Args:
            sort_criterion (str): Criterion to sort by (price, rating, etc.)
            
        Returns:
            str: Sorted product list
        """
        if not sort_criterion:
            return "Please specify how you'd like to sort the products (e.g., by price, rating, popularity)."
        
        sort_criterion = sort_criterion.lower()
        
        # Determine which products to sort
        products_to_sort = []
        if self.current_category:
            products_to_sort = [p for p in self.products if p['category'] == self.current_category]
            context = f"products in {self.current_category}"
        else:
            # Use recently viewed category or all products
            if self.recently_viewed:
                last_viewed_category = self.recently_viewed[-1]['category']
                products_to_sort = [p for p in self.products if p['category'] == last_viewed_category]
                context = f"products in {last_viewed_category}"
            else:
                products_to_sort = self.products
                context = "all products"
        
        # Sort based on criterion
        if 'price' in sort_criterion:
            if 'high' in sort_criterion or 'expensive' in sort_criterion or 'descending' in sort_criterion:
                sorted_products = sorted(products_to_sort, key=lambda x: x['price'], reverse=True)
                sort_desc = "price (highest first)"
            else:
                sorted_products = sorted(products_to_sort, key=lambda x: x['price'])
                sort_desc = "price (lowest first)"
        elif 'rating' in sort_criterion or 'review' in sort_criterion or 'best' in sort_criterion:
            sorted_products = sorted(products_to_sort, key=lambda x: x['rating'], reverse=True)
            sort_desc = "customer rating (highest first)"
        elif 'popular' in sort_criterion or 'top' in sort_criterion:
            sorted_products = sorted(products_to_sort, key=lambda x: x['popular'], reverse=True)
            sort_desc = "popular product (highest first)"
          
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
