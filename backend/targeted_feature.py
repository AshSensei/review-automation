import json
import re
import os
from collections import Counter
from bs4 import BeautifulSoup
from transformers import pipeline
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import nltk
from openai import OpenAI
import time
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Configuration and Setup ---

# Load environment variables from .env file
load_dotenv()

# Configure logging to provide informative output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes, allowing React frontend to connect

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpus/stopwords')
except LookupError:
    logger.info("Downloading NLTK data (punkt, stopwords)...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# --- Core Analysis Class ---

class OpenAIHybridReviewAnalyzer:
    """
    A sophisticated review analyzer that uses a hybrid approach:
    - Local transformer models for fast, bulk sentiment analysis.
    - OpenAI's GPT models for nuanced theme extraction, issue identification, and strategic insights.
    - Optional keyword filtering to focus analysis on specific product parts/aspects.
    """
    def __init__(self, openai_api_key: str):
        """Initialize the hybrid analyzer with the OpenAI API key."""
        self.device = -1  # Use -1 for CPU, or 0 for GPU if available
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Initialize local models for bulk processing
        try:
            logger.info("Loading local sentiment analysis model...")
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=self.device
            )
            logger.info(f"Local models loaded successfully on device: {self.device}")
        except Exception as e:
            logger.error(f"Error loading local RoBERTa model: {e}. Falling back to default.")
            self.sentiment_analyzer = pipeline("sentiment-analysis", device=self.device)
        
        self.token_usage = {'total_tokens': 0, 'estimated_cost_usd': 0.0}

        # Predefined keyword mappings for common product parts/features
        self.predefined_keywords = {
            'battery': ['battery', 'power', 'charge', 'charging', 'juice', 'life', 'drain', 'lasted', 'lasts'],
            'screen': ['screen', 'display', 'monitor', 'lcd', 'oled', 'brightness', 'resolution', 'pixel', 'clarity'],
            'camera': ['camera', 'photo', 'picture', 'video', 'lens', 'zoom', 'focus', 'quality', 'megapixel'],
            'speaker': ['speaker', 'sound', 'audio', 'volume', 'bass', 'treble', 'music', 'noise', 'clarity'],
            'keyboard': ['keyboard', 'keys', 'typing', 'key', 'backlight', 'tactile', 'mechanical', 'responsive'],
            'button': ['button', 'buttons', 'click', 'press', 'responsive', 'clicky', 'tactile', 'feedback'],
            'build': ['build', 'construction', 'material', 'plastic', 'metal', 'quality', 'solid', 'sturdy'],
            'design': ['design', 'look', 'appearance', 'aesthetic', 'style', 'beautiful', 'ugly', 'attractive'],
            'performance': ['performance', 'speed', 'fast', 'slow', 'lag', 'responsive', 'smooth', 'sluggish'],
            'connectivity': ['wifi', 'bluetooth', 'connection', 'connect', 'signal', 'pairing', 'wireless'],
            'comfort': ['comfort', 'ergonomic', 'fit', 'comfortable', 'grip', 'wear', 'fatigue', 'strain'],
            'durability': ['durable', 'sturdy', 'break', 'broken', 'fragile', 'solid', 'last', 'lasting'],
            'weight': ['weight', 'heavy', 'light', 'lightweight', 'portable', 'bulk', 'hefty'],
            'size': ['size', 'compact', 'large', 'small', 'dimensions', 'footprint', 'big', 'tiny'],
            'price': ['price', 'cost', 'expensive', 'cheap', 'value', 'worth', 'affordable', 'overpriced'],
            'microphone': ['microphone', 'mic', 'audio input', 'voice', 'recording', 'sound quality', 'pickup', 'clarity', 'mute', 'voice chat'],
            'ergonomics': ['ergonomic', 'comfort', 'grip', 'hand', 'wrist', 'strain', 'comfortable', 'fit'],
            'software': ['software', 'app', 'driver', 'firmware', 'update', 'bug', 'glitch', 'interface'],
            'charging': ['charging', 'charge', 'usb', 'cable', 'port', 'power', 'adapter', 'fast charge']
        }

    def _update_token_usage(self, usage):
        """Update the token usage and estimated cost."""
        if not usage: 
            return
        total_tokens = usage.total_tokens
        cost = total_tokens * (0.375 / 1_000_000)  # Approx. cost for gpt-4o-mini
        self.token_usage['total_tokens'] += total_tokens
        self.token_usage['estimated_cost_usd'] += cost
        logger.info(f"OpenAI API call used {total_tokens} tokens. Cumulative total: {self.token_usage['total_tokens']}. Estimated cost: ${self.token_usage['estimated_cost_usd']:.4f}")

    def batch_sentiment_analysis(self, reviews: List[Dict]) -> List[Dict]:
        """Perform bulk sentiment analysis using local models."""
        logger.info("📊 Running batch sentiment analysis...")
        sentiments = []
        for i, review in enumerate(reviews):
            text = review.get('review_text', '')
            if text and text.strip():
                try:
                    truncated_text = text[:512]
                    result = self.sentiment_analyzer(truncated_text)[0]
                    # The model returns 'LABEL_0', 'LABEL_1', 'LABEL_2' or 'POSITIVE', 'NEGATIVE', 'NEUTRAL'
                    # Standardize the labels
                    label = result['label'].upper()
                    if 'POSITIVE' in label or 'LABEL_2' in label:
                        sentiment = 'POSITIVE'
                    elif 'NEGATIVE' in label or 'LABEL_0' in label:
                        sentiment = 'NEGATIVE'
                    else:
                        sentiment = 'NEUTRAL'
                    
                    sentiments.append({
                        'review_index': i, 
                        'sentiment': sentiment, 
                        'confidence': result['score'],
                        'review_rating': review.get('review_rating', 3)
                    })
                except Exception as e:
                    logger.warning(f"Sentiment analysis failed for review {i}: {e}")
                    sentiments.append({
                        'review_index': i, 
                        'sentiment': 'NEUTRAL', 
                        'confidence': 0.5,
                        'review_rating': review.get('review_rating', 3)
                    })
        return sentiments

    def filter_reviews_for_target_part(self, reviews: List[Dict], target_part: str, use_llm_keywords: bool = False) -> List[Dict]:
        """
        Filter reviews to find those that specifically mention the target part/aspect.
        
        Args:
            reviews: List of review dictionaries
            target_part: The target part/aspect to filter for
            use_llm_keywords: If True, use LLM to generate keywords for custom/unknown parts
        """
        if not target_part:
            return reviews
            
        logger.info(f"🎯 Filtering reviews for target part: '{target_part}' (LLM keywords: {use_llm_keywords})")
        
        # Generate keyword variations for the target part
        target_keywords = self._generate_target_keywords(target_part, use_llm_keywords)
        
        relevant_reviews = []
        for review in reviews:
            text = review.get('review_text', '').lower()
            if any(keyword in text for keyword in target_keywords):
                relevant_reviews.append(review)
        
        logger.info(f"Found {len(relevant_reviews)} reviews mentioning '{target_part}' out of {len(reviews)} total reviews")
        return relevant_reviews

    def _generate_target_keywords(self, target_part: str, use_llm: bool = False) -> List[str]:
        """
        Generate relevant keywords for the target part to improve filtering.
        
        Args:
            target_part: The target part/aspect
            use_llm: If True, use LLM to generate keywords for custom parts
        """
        target_part_lower = target_part.lower().strip()
        
        # First, check if we have predefined keywords for this part
        for predefined_part, keywords in self.predefined_keywords.items():
            if predefined_part in target_part_lower or target_part_lower in predefined_part:
                logger.info(f"Using predefined keywords for '{target_part}': {keywords}")
                return self._add_variations(keywords)
        
        # If not found in predefined and LLM is requested, use LLM to generate keywords
        if use_llm:
            logger.info(f"Generating keywords for '{target_part}' using LLM...")
            llm_keywords = self._generate_keywords_with_llm(target_part)
            if llm_keywords:
                return self._add_variations(llm_keywords)
        
        # Fallback: use the target part itself with basic variations
        logger.info(f"Using fallback keywords for '{target_part}'")
        base_keywords = [target_part_lower]
        return self._add_variations(base_keywords)

    def _generate_keywords_with_llm(self, target_part: str) -> List[str]:
        """Use LLM to generate relevant keywords for a custom target part."""
        prompt = f"""
        For the product feature/component "{target_part}", generate 8-12 relevant keywords that customers might use in reviews when discussing this aspect.
        
        Include:
        - The exact term and close variations
        - Common synonyms and related terms
        - Casual/informal ways people might refer to it
        - Performance-related terms (good/bad aspects)
        
        Return a JSON object with a single key "keywords" containing an array of lowercase strings.
        
        Example for "microphone": {{"keywords": ["microphone", "mic", "audio input", "voice", "recording", "sound quality", "pickup", "clarity", "mute", "voice chat"]}}
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=300
            )
            self._update_token_usage(response.usage)
            
            result = json.loads(response.choices[0].message.content)
            keywords = result.get('keywords', [])
            logger.info(f"LLM generated {len(keywords)} keywords for '{target_part}': {keywords}")
            return keywords
        except Exception as e:
            logger.error(f"Failed to generate keywords with LLM for '{target_part}': {e}")
            return []

    def _add_variations(self, base_keywords: List[str]) -> List[str]:
        """Add plural/singular and other variations to the keyword list."""
        extended_keywords = []
        for keyword in base_keywords:
            keyword = keyword.lower().strip()
            extended_keywords.append(keyword)
            
            # Add plural/singular variations
            if keyword.endswith('s') and len(keyword) > 3:
                extended_keywords.append(keyword[:-1])  # Remove 's'
            elif not keyword.endswith('s'):
                extended_keywords.append(keyword + 's')  # Add 's'
            
            # Add some common variations
            if keyword.endswith('y'):
                extended_keywords.append(keyword[:-1] + 'ies')  # battery -> batteries
        
        # Remove duplicates and return
        return list(set(extended_keywords))

    def extract_issues_with_llm(self, reviews: List[Dict], product_type: str, target_part: Optional[str] = None, use_llm_keywords: bool = False) -> List[Dict]:
        """Use OpenAI's GPT to extract specific, actionable issues from negative reviews, optionally focusing on a target part."""
        focus_text = f" specifically related to the '{target_part}'" if target_part else ""
        logger.info(f"🤖 Extracting issues from negative reviews{focus_text} with LLM...")
        
        # Filter for target part if specified
        if target_part:
            reviews = self.filter_reviews_for_target_part(reviews, target_part, use_llm_keywords)

        # Filter negative reviews (rating <= 2 or negative sentiment)
        negative_reviews = []
        for review in reviews:
            rating = review.get('review_rating', 3)
            if rating <= 2:
                negative_reviews.append(review)
            elif rating == 3:  # Check sentiment for 3-star reviews
                try:
                    sentiment = self.sentiment_analyzer(review['review_text'][:512])[0]
                    if 'neg' in sentiment['label'].lower() and sentiment['score'] > 0.6:
                        negative_reviews.append(review)
                except:
                    pass

        if not negative_reviews:
            logger.info(f"No significant negative reviews found for {product_type}{focus_text} to extract issues from.")
            return []
        
        # Limit to 20 most informative negative reviews
        negative_reviews = negative_reviews[:20]
        
        # Prepare review text for LLM
        review_texts = []
        for i, review in enumerate(negative_reviews):
            rating = review.get('review_rating', 'N/A')
            text = review['review_text'][:500]  # Limit length
            review_texts.append(f"Review {i+1} (Rating: {rating}): {text}")
        
        combined_text = "\n\n".join(review_texts)
        
        target_instruction = f"\nFOCUS SPECIFICALLY ON ISSUES RELATED TO: {target_part.upper()}. Only extract issues that directly relate to this component/aspect." if target_part else ""
        
        prompt = f"""Analyze these negative {product_type} reviews and extract specific product issues.
{target_instruction}

For each issue found, provide:
1. Issue name (brief, specific)
2. Description (what exactly is wrong)
3. Frequency (how many reviews mention this)
4. Severity (high/medium/low based on impact)
5. Example quote from reviews

Focus on concrete, actionable issues, not vague complaints.

Reviews:
{combined_text}

Return as JSON array with format:
[
  {{
    "issue_name": "connection_drops",
    "description": "Device frequently disconnects during use",
    "frequency": 3,
    "severity": "high",
    "example_quote": "keeps disconnecting every few minutes"
  }}
]"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            
            self._update_token_usage(response.usage)

            # Get the response content and strip markdown
            response_content = response.choices[0].message.content.strip()
            if response_content.startswith("```json"):
                response_content = response_content[7:-3].strip()

            issues = json.loads(response_content)
            
            # Add review context
            for issue in issues:
                issue['type'] = self.categorize_issue(issue['issue_name'])
                issue['review_count'] = len(negative_reviews)
                if target_part:
                    issue['target_part'] = target_part
            
            return issues
            
        except Exception as e:
            logger.error(f"LLM issue extraction failed: {e}")
            return self.fallback_issue_extraction(negative_reviews, target_part)

    def categorize_issue(self, issue_name: str) -> str:
        """Categorize an issue into a predefined type based on keywords."""
        issue_name = issue_name.lower()
        categories = {
            'connectivity': ['connection', 'disconnect', 'bluetooth', 'wifi', 'pairing'],
            'build_quality': ['build', 'quality', 'cheap', 'flimsy', 'material', 'plastic'],
            'performance': ['slow', 'lag', 'performance', 'speed', 'responsive', 'crash'],
            'battery': ['battery', 'charge', 'power', 'drain', 'life'],
            'durability': ['break', 'broke', 'broken', 'durable', 'last', 'fail'],
            'comfort': ['comfort', 'hurt', 'pain', 'ergonomic', 'fit'],
            'functionality': ['work', 'function', 'feature', 'button', 'software', 'bug']
        }
        for category, keywords in categories.items():
            if any(keyword in issue_name for keyword in keywords):
                return category
        return 'other'

    def analyze_themes_with_llm(self, reviews: List[Dict], product_type: str, target_part: Optional[str] = None, use_llm_keywords: bool = False) -> Dict[str, Any]:
        """Analyze overarching themes and their sentiment using a smart sample of reviews, optionally focusing on a target part."""
        focus_text = f" specifically related to the '{target_part}'" if target_part else ""
        logger.info(f"🎯 Analyzing themes{focus_text} with LLM...")
        
        if not reviews: 
            return {'themes': {}, 'discovered_themes': [], 'sample_size': 0, 'target_part': target_part}

        # Filter for target part if specified
        original_count = len(reviews)
        if target_part:
            reviews = self.filter_reviews_for_target_part(reviews, target_part, use_llm_keywords)
            if not reviews:
                logger.warning(f"No reviews found mentioning '{target_part}'")
                return {'themes': {}, 'discovered_themes': [], 'sample_size': 0, 'target_part': target_part}

        sample_reviews = self._get_representative_sample(reviews, 15)
        sample_texts = [f"Review (Rating: {r.get('review_rating', 'N/A')}): {r.get('review_text', '')[:400]}" for r in sample_reviews]
        combined_text = "\n\n".join(sample_texts)
        
        target_instruction = f"\nFOCUS SPECIFICALLY ON THEMES RELATED TO: {target_part.upper()}. Only identify themes that directly relate to this component/aspect." if target_part else ""
        
        # Two-step process: 1) Discover themes, 2) Analyze sentiment
        
        # Step 1: Theme Discovery
        theme_prompt = f"""Analyze these {product_type} reviews and identify the main themes/aspects customers discuss{focus_text}.

Return 8-10 specific themes relevant to this product type. Focus on concrete features and aspects, not general sentiment.
{target_instruction}

Reviews:
{combined_text}

Return as JSON array of theme names:
["theme1", "theme2", "theme3", ...]

Example for gaming controller: ["button_responsiveness", "build_quality", "battery_life", "connectivity", "ergonomics"]"""

        try:
            theme_response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": theme_prompt}],
                temperature=0.2,
                max_tokens=500
            )
            
            # Get the response content and strip markdown
            response_content = theme_response.choices[0].message.content.strip()
            if response_content.startswith("```json"):
                response_content = response_content[7:-3].strip()

            themes = json.loads(response_content)

            # Step 2: Analyze sentiment for each theme
            sentiment_prompt = f"""For each theme, analyze the sentiment in these {product_type} reviews{focus_text}:

Themes to analyze: {themes}

Reviews:
{combined_text}

For each theme, provide:
1. Number of mentions
2. Overall sentiment (positive/negative/neutral)
3. Confidence score (0-1)
4. Key phrases/examples

Return as JSON object:
{{
  "theme_name": {{
    "mentions": 5,
    "sentiment": "positive",
    "confidence": 0.8,
    "key_phrases": ["works great", "very responsive"],
    "example_quote": "buttons are very responsive"
  }}
}}"""

            sentiment_response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": sentiment_prompt}],
                temperature=0.1,
                max_tokens=1500
            )
            
            # Track token usage
            total_tokens = theme_response.usage.total_tokens + sentiment_response.usage.total_tokens
            self.token_usage['total_tokens'] += total_tokens
            self.token_usage['estimated_cost_usd'] += total_tokens * 0.375 / 1_000_000

            # Get the response content and strip markdown
            response_content = sentiment_response.choices[0].message.content.strip()
            if response_content.startswith("```json"):
                response_content = response_content[7:-3].strip()
                
            theme_analysis = json.loads(response_content)
            
            return {
                'themes': theme_analysis,
                'discovered_themes': themes,
                'sample_size': len(sample_reviews),
                'target_part': target_part,
                'filtered_review_count': len(reviews) if target_part else None,
                'original_review_count': original_count
            }
            
        except Exception as e:
            logger.error(f"LLM theme analysis failed: {e}")
            return self.fallback_theme_analysis(reviews, product_type, target_part)

    def _get_representative_sample(self, reviews: List[Dict], target_count: int) -> List[Dict]:
        """Create a diverse sample of reviews based on rating and length."""
        if len(reviews) <= target_count: 
            return reviews
        
        # Sort by length to get more informative reviews first
        reviews.sort(key=lambda r: len(r.get('review_text', '')), reverse=True)
        
        # Categorize by rating
        high_rated = [r for r in reviews if r.get('review_rating', 3) >= 4]
        low_rated = [r for r in reviews if r.get('review_rating', 3) <= 2]
        mid_rated = [r for r in reviews if r.get('review_rating', 3) == 3]
        
        # Distribute sample across rating categories
        sample = []
        seen = set()
        
        # Get mix of ratings
        for pool in [high_rated, low_rated, mid_rated]:
            for review in pool:
                if len(sample) < target_count and id(review) not in seen:
                    sample.append(review)
                    seen.add(id(review))
        
        # Fill remaining with any reviews
        remaining = target_count - len(sample)
        if remaining > 0:
            other_reviews = [r for r in reviews if id(r) not in seen]
            sample.extend(other_reviews[:remaining])
        
        return sample

    def generate_insights_with_llm(self, themes: Dict, issues: List[Dict], metrics: Dict, product_type: str, target_part: Optional[str] = None) -> Dict:
        """Generate executive summary and recommendations using LLM."""
        focus_text = f" focusing on '{target_part}'" if target_part else ""
        logger.info(f"💡 Generating insights{focus_text} with LLM...")
        
        # Prepare context for LLM
        context = f"""Product Analysis Context:
Product Type: {product_type}
Target Focus: {target_part if target_part else 'General Analysis'}
Total Reviews: {metrics['total_reviews']}
Average Rating: {metrics['average_rating']:.1f}/5
Sentiment Distribution: {metrics['sentiment_distribution']}

Top Themes:
{json.dumps(themes.get('themes', {}), indent=2)}

Key Issues:
{json.dumps(issues[:5], indent=2)}"""

        target_instruction = f"\nIMPORTANT: This analysis is specifically focused on the '{target_part}' aspect of the product. Tailor all insights and recommendations to this specific component/feature." if target_part else ""

        prompt = f"""Based on this product analysis{focus_text}, provide:

1. Executive summary (2-3 sentences highlighting key findings)
2. Top 5 actionable recommendations for improvement
3. Priority level for each recommendation (high/medium/low)
4. Business impact assessment

{target_instruction}

{context}

Return as JSON:
{{
  "executive_summary": "...",
  "recommendations": [
    {{
      "recommendation": "...",
      "priority": "high",
      "impact": "...",
      "rationale": "..."
    }}
  ],
  "key_insights": ["insight1", "insight2", ...]
}}"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            
            self._update_token_usage(response.usage)
        
            # Get the response content and strip markdown
            response_content = response.choices[0].message.content.strip()
            if response_content.startswith("```json"):
                response_content = response_content[7:-3].strip()

            return json.loads(response_content)
            
        except Exception as e:
            logger.error(f"LLM insights generation failed: {e}")
            return {
                "executive_summary": f"Analysis completed for {product_type}{focus_text} with mixed results requiring attention.",
                "recommendations": [{"recommendation": "Review customer feedback patterns", "priority": "medium", "impact": "moderate"}],
                "key_insights": ["Customer feedback analysis completed"]
            }

    def calculate_enhanced_metrics(self, reviews: List[Dict], sentiments: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive metrics combining local and LLM analysis."""
        if not reviews:
            return {}
            
        ratings = [r.get('review_rating', 3) for r in reviews if r.get('review_rating') is not None]
        review_lengths = [len(r['review_text']) for r in reviews if r['review_text'].strip()]
        
        # Sentiment distribution from local analysis
        sentiment_counts = Counter(s['sentiment'] for s in sentiments)
        
        # Additional metrics
        confidence_scores = [s['confidence'] for s in sentiments]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        return {
            'total_reviews': len(reviews),
            'average_rating': round(sum(ratings) / len(ratings), 2) if ratings else 0,
            'rating_distribution': dict(Counter(ratings)),
            'average_review_length': round(sum(review_lengths) / len(review_lengths)) if review_lengths else 0,
            'sentiment_distribution': dict(sentiment_counts),
            'average_sentiment_confidence': round(avg_confidence, 3),
            'analysis_quality': 'enhanced' if avg_confidence > 0.7 else 'standard'
        }

    def fallback_issue_extraction(self, reviews: List[Dict], target_part: Optional[str] = None) -> List[Dict]:
        """Fallback issue extraction using pattern matching."""
        logger.info("Using fallback issue extraction")
        
        issue_patterns = [
            (r"(break|broke|broken|fail|failed|stop|stopped)\s+(\w+)", "failure"),
            (r"(doesn't|don't|won't|can't)\s+(\w+)", "functionality"),
            (r"(poor|bad|terrible|awful|horrible)\s+(\w+)", "quality"),
            (r"(disconnect|connection|connectivity)", "connectivity")
        ]
        
        issues = []
        for review in reviews:
            text = review['review_text'].lower()
            for pattern, issue_type in issue_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    if isinstance(match, tuple):
                        issue_description = f"{match[0]} {match[1]}"
                    else:
                        issue_description = match
                    
                    issue_dict = {
                        'issue_name': issue_description,
                        'description': f"Pattern match: {issue_description}",
                        'frequency': 1,
                        'severity': 'medium',
                        'example_quote': text[:100] + "...",
                        'type': issue_type
                    }
                    
                    if target_part:
                        issue_dict['target_part'] = target_part
                        
                    issues.append(issue_dict)
        
        # Aggregate duplicates
        issue_counts = Counter(issue['issue_name'] for issue in issues)
        unique_issues = []
        for issue in issues:
            if issue['issue_name'] not in [ui['issue_name'] for ui in unique_issues]:
                issue['frequency'] = issue_counts[issue['issue_name']]
                unique_issues.append(issue)
        
        return sorted(unique_issues, key=lambda x: x['frequency'], reverse=True)[:5]

    def generate_comprehensive_analysis(self, reviews: List[Dict], product_type: str = "product", target_part: Optional[str] = None, use_llm_keywords: bool = False) -> Dict[str, Any]:
            """Generate comprehensive analysis using hybrid approach."""
            
            logger.info(f"🚀 Starting hybrid analysis for {len(reviews)} reviews...")
            start_time = time.time()
            
            # Step 1: Local sentiment analysis (free/fast)
            sentiments = self.batch_sentiment_analysis(reviews)
            
            # Step 2: LLM-powered issue extraction (focused on negatives)
            issues = self.extract_issues_with_llm(reviews, product_type, target_part, use_llm_keywords)
            
            # Step 3: LLM-powered theme analysis (smart sampling)
            themes = self.analyze_themes_with_llm(reviews, product_type, target_part, use_llm_keywords)
            
            # Step 4: Enhanced metrics calculation
            metrics = self.calculate_enhanced_metrics(reviews, sentiments)
            
            # Step 5: LLM-powered insights generation
            insights = self.generate_insights_with_llm(themes, issues, metrics, product_type, target_part)
            
            analysis_time = time.time() - start_time
            
            return {
                'themes': themes,
                'issues': issues,
                'metrics': metrics,
                'insights': insights,
                'analysis_metadata': {
                    'total_reviews': len(reviews),
                    'analysis_date': datetime.now().isoformat(),
                    'product_type': product_type,
                    'target_part': target_part,
                    'model_used': 'gpt-4o-mini + local models',
                    'analysis_time_seconds': round(analysis_time, 2),
                    'token_usage': self.token_usage.copy()
                }
            }

    def create_detailed_report(self, analysis_results: Dict[str, Any]) -> str:
        """Create a detailed, actionable report in plain text."""
        
        themes = analysis_results['themes']
        issues = analysis_results['issues']
        metrics = analysis_results['metrics']
        insights = analysis_results['insights']
        metadata = analysis_results['analysis_metadata']
        
        lines = []
        lines.append("=" * 80)
        lines.append("AI-ENHANCED PRODUCT REVIEW ANALYSIS")
        lines.append("=" * 80)
        lines.append(f"Product: {metadata['product_type'].title()}")
        if metadata.get('target_part'):
            lines.append(f"Focus Area: {metadata['target_part'].title()}")
        lines.append(f"Reviews Analyzed: {metadata['total_reviews']}")
        lines.append(f"Analysis Date: {metadata['analysis_date']}")
        lines.append(f"Model: {metadata['model_used']}")
        lines.append(f"Processing Time: {metadata['analysis_time_seconds']}s")
        lines.append(f"API Cost: ${metadata['token_usage']['estimated_cost_usd']:.4f}")
        lines.append("")
        
        # Executive Summary
        lines.append("EXECUTIVE SUMMARY")
        lines.append("-" * 40)
        lines.append(insights.get('executive_summary', 'Analysis completed.'))
        lines.append("")
        
        # Key Insights
        if insights.get('key_insights'):
            lines.append("KEY INSIGHTS")
            lines.append("-" * 40)
            for insight in insights['key_insights']:
                lines.append(f"- {insight}")
            lines.append("")
        
        # Key Metrics
        lines.append("KEY METRICS")
        lines.append("-" * 40)
        lines.append(f"Average Rating: {metrics['average_rating']:.1f}/5")
        lines.append(f"Total Reviews: {metrics['total_reviews']}")
        lines.append(f"Average Review Length: {metrics['average_review_length']:.0f} characters")
        lines.append(f"Analysis Quality: {metrics['analysis_quality'].title()}")
        lines.append("")
        
        # Sentiment Distribution
        lines.append("Sentiment Distribution:")
        total_sentiment = sum(metrics['sentiment_distribution'].values())
        for sentiment, count in metrics['sentiment_distribution'].items():
            percentage = (count / total_sentiment * 100) if total_sentiment > 0 else 0
            lines.append(f"  {sentiment.title()}: {count} ({percentage:.1f}%)")
        lines.append("")
        
        # Top Issues (LLM-powered)
        lines.append("TOP ISSUES IDENTIFIED")
        lines.append("-" * 40)
        if issues:
            for i, issue in enumerate(issues[:5], 1):
                lines.append(f"{i}. {issue['issue_name'].replace('_', ' ').title()}")
                lines.append(f"   Description: {issue['description']}")
                lines.append(f"   Frequency: {issue['frequency']} mention(s)")
                lines.append(f"   Severity: {issue['severity'].title()}")
                lines.append(f"   Example: \"{issue['example_quote']}\"")
                if issue.get('target_part'):
                    lines.append(f"   Focus Area: {issue['target_part']}")
                lines.append("")
        else:
            lines.append("No significant issues identified.")
            lines.append("")
        
        # Theme Analysis (LLM-powered)
        lines.append("THEME ANALYSIS")
        lines.append("-" * 40)
        
        if themes.get('themes'):
            if themes.get('target_part'):
                lines.append(f"Focus Area: {themes['target_part']}")
                if themes.get('filtered_review_count'):
                    lines.append(f"Relevant Reviews: {themes['filtered_review_count']} of {themes.get('original_review_count', 'N/A')}")
                lines.append("")
            
            for theme_name, theme_data in themes['themes'].items():
                lines.append(f"{theme_name.replace('_', ' ').title()}")
                lines.append(f"   Mentions: {theme_data['mentions']}")
                lines.append(f"   Sentiment: {theme_data['sentiment'].title()}")
                lines.append(f"   Confidence: {theme_data['confidence']:.2f}")
                
                if theme_data.get('key_phrases'):
                    lines.append(f"   Key Phrases: {', '.join(theme_data['key_phrases'])}")
                
                if theme_data.get('example_quote'):
                    lines.append(f"   Example: \"{theme_data['example_quote']}\"")
                lines.append("")
        
        # Recommendations (LLM-powered)
        lines.append("AI-POWERED RECOMMENDATIONS")
        lines.append("-" * 40)
        
        if insights.get('recommendations'):
            for i, rec in enumerate(insights['recommendations'], 1):
                lines.append(f"{i}. {rec['recommendation']}")
                lines.append(f"   Priority: {rec['priority'].title()}")
                lines.append(f"   Impact: {rec['impact']}")
                lines.append(f"   Rationale: {rec['rationale']}")
                lines.append("")
        
        # Analysis Statistics
        lines.append("ANALYSIS STATISTICS")
        lines.append("-" * 40)
        lines.append(f"Tokens Used: {metadata['token_usage']['total_tokens']:,}")
        lines.append(f"Estimated Cost: ${metadata['token_usage']['estimated_cost_usd']:.4f}")
        lines.append(f"Processing Time: {metadata['analysis_time_seconds']}s")
        lines.append("")
        
        lines.append("=" * 80)
        lines.append("Analysis Complete")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    def fallback_theme_analysis(self, reviews: List[Dict], product_type: str, target_part: Optional[str] = None) -> Dict[str, Any]:
        """Complete the fallback theme analysis using keyword matching."""
        logger.info("Using fallback theme analysis")
        
        # Default themes by product type
        default_themes = {
            'gaming controller': ['build_quality', 'battery_life', 'connectivity', 'comfort', 'performance'],
            'headphones': ['sound_quality', 'comfort', 'build_quality', 'battery_life', 'connectivity'],
            'keyboard': ['key_responsiveness', 'build_quality', 'comfort', 'backlighting', 'software'],
            'mouse': ['precision', 'comfort', 'build_quality', 'battery_life', 'software'],
            'default': ['quality', 'performance', 'design', 'price', 'durability']
        }
        
        themes = default_themes.get(product_type.lower(), default_themes['default'])
        
        # Basic keyword matching for themes
        theme_analysis = {}
        for theme in themes:
            theme_keywords = theme.split('_')  # Simple keyword extraction
            mentions = 0
            positive_mentions = 0
            example_quotes = []
            
            for review in reviews:
                text = review['review_text'].lower()
                if any(keyword in text for keyword in theme_keywords):
                    mentions += 1
                    if review.get('review_rating', 3) >= 4:
                        positive_mentions += 1
                    if len(example_quotes) < 1:  # Get one example
                        example_quotes.append(review['review_text'][:100] + "...")
            
            # Determine sentiment based on ratings
            if mentions > 0:
                sentiment_ratio = positive_mentions / mentions
                if sentiment_ratio > 0.6:
                    sentiment = 'positive'
                elif sentiment_ratio < 0.4:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
            else:
                sentiment = 'neutral'
            
            theme_analysis[theme] = {
                'mentions': mentions,
                'sentiment': sentiment,
                'confidence': 0.5,  # Lower confidence for fallback
                'key_phrases': theme_keywords,
                'example_quote': example_quotes[0] if example_quotes else ''
            }
        
        return {
            'themes': theme_analysis,
            'discovered_themes': themes,
            'sample_size': len(reviews),
            'target_part': target_part
        }
# HTML parsing functions
def parse_reviews_from_file(file_path: str) -> Optional[List[Dict]]:
    """Parse reviews from HTML file with enhanced error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return None
    
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        reviews = []
        
        # Try multiple selectors
        selectors = ['li.review-item', '.review-item', '.review', '[data-testid="review"]']
        
        review_elements = []
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                review_elements = elements
                break
        
        if not review_elements:
            logger.warning("No review elements found")
            return []
        
        for element in review_elements:
            review_data = {'review_text': '', 'review_rating': None}
            
            # Extract text
            text_selectors = ['p.pre-white-space', '.review-text', 'p', '.content']
            for selector in text_selectors:
                text_element = element.select_one(selector)
                if text_element:
                    review_data['review_text'] = text_element.get_text(strip=True)
                    break
            
            # Extract rating
            rating_element = element.select_one('p.visually-hidden')
            if rating_element:
                rating_match = re.search(r'Rated (\d(?:\.\d)?)', rating_element.text)
                if rating_match:
                    review_data['review_rating'] = float(rating_match.group(1))
            
            if review_data['review_text']:
                reviews.append(review_data)
        
        logger.info(f"Successfully parsed {len(reviews)} reviews")
        return reviews
    
    except Exception as e:
        logger.error(f"Error parsing HTML: {e}")
        return None

def parse_reviews_from_html_string(html_str: str) -> Optional[List[Dict]]:
    """Parse reviews from HTML string with enhanced error handling."""
    try:
        soup = BeautifulSoup(html_str, 'html.parser')
        reviews = []
        
        # Try multiple selectors
        selectors = ['li.review-item', '.review-item', '.review', '[data-testid="review"]']
        
        review_elements = []
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                review_elements = elements
                break
        
        if not review_elements:
            logger.warning("No review elements found")
            return []
        
        for element in review_elements:
            review_data = {'review_text': '', 'review_rating': None}
            
            # Extract text
            text_selectors = ['p.pre-white-space', '.review-text', 'p', '.content']
            for selector in text_selectors:
                text_element = element.select_one(selector)
                if text_element:
                    review_data['review_text'] = text_element.get_text(strip=True)
                    break
            
            # Extract rating
            rating_element = element.select_one('p.visually-hidden')
            if rating_element:
                rating_match = re.search(r'Rated (\d(?:\.\d)?)', rating_element.text)
                if rating_match:
                    review_data['review_rating'] = float(rating_match.group(1))
            
            if review_data['review_text']:
                reviews.append(review_data)
        
        logger.info(f"Successfully parsed {len(reviews)} reviews")
        return reviews
    
    except Exception as e:
        logger.error(f"Error parsing HTML: {e}")
        return None

# Flask API endpoints
@app.route('/api/analyze', methods=['POST'])
def analyze_reviews():
    """Main analysis endpoint for JSON review data."""
    try:
        data = request.get_json()
        
        # Extract parameters
        reviews = data.get('reviews', [])
        product_type = data.get('product_type', 'product')
        target_part = data.get('target_part')
        use_llm_keywords = data.get('use_llm_keywords', False)
        
        if not reviews:
            return jsonify({'error': 'No reviews provided'}), 400
        
        # Initialize analyzer
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            return jsonify({'error': 'OpenAI API key not configured'}), 500
        
        analyzer = OpenAIHybridReviewAnalyzer(openai_api_key)
        
        # Generate analysis
        analysis_results = analyzer.generate_comprehensive_analysis(
            reviews, product_type, target_part, use_llm_keywords
        )
        
        return jsonify(analysis_results)
        
    except Exception as e:
        logger.error(f"Analysis API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-html', methods=['POST'])
def analyze_html():
    """Analysis endpoint for HTML review data."""
    try:
        data = request.get_json()
        html_content = data.get('html', '')
        product_type = data.get('product_type', 'product')
        target_part = data.get('target_part')
        use_llm_keywords = data.get('use_llm_keywords', False)
        
        if not html_content:
            return jsonify({'error': 'No HTML content provided'}), 400
        
        # Parse reviews from HTML
        reviews = parse_reviews_from_html_string(html_content)
        if not reviews:
            return jsonify({'error': 'No reviews found in HTML'}), 400
        
        # Initialize analyzer
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            return jsonify({'error': 'OpenAI API key not configured'}), 500
        
        analyzer = OpenAIHybridReviewAnalyzer(openai_api_key)
        
        # Generate analysis
        analysis_results = analyzer.generate_comprehensive_analysis(
            reviews, product_type, target_part, use_llm_keywords
        )
        
        return jsonify(analysis_results)
        
    except Exception as e:
        logger.error(f"HTML analysis API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'openai_configured': bool(os.getenv('OPENAI_API_KEY'))
    })

# Main execution function
def main():
    """Main execution function for command-line usage."""
    try:
        # Get OpenAI API key
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            openai_api_key = input("Enter your OpenAI API key: ").strip()
        
        if not openai_api_key:
            print("OpenAI API key is required.")
            return
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except:
            pass
        
        # Parse reviews (update with your file path)
        reviews = parse_reviews_from_file('sample.html')
        if not reviews:
            print("No reviews found or error parsing file.")
            return
        
        print(f"Found {len(reviews)} reviews to analyze...")
        
        # Get analysis parameters
        product_type = input("Enter product type (e.g., 'gaming controller', 'headphones'): ").strip()
        if not product_type:
            product_type = "product"
        
        target_part = input("Enter target part to focus on (optional, press Enter to skip): ").strip()
        if not target_part:
            target_part = None
        
        use_llm_keywords = False
        if target_part:
            llm_choice = input("Use LLM to generate keywords for custom parts? (y/n): ").strip().lower()
            use_llm_keywords = llm_choice == 'y'
        
        # Initialize analyzer
        print("Initializing AI-enhanced analyzer...")
        analyzer = OpenAIHybridReviewAnalyzer(openai_api_key)
        
        # Generate analysis
        print("Running AI-enhanced analysis...")
        analysis_results = analyzer.generate_comprehensive_analysis(
            reviews, product_type, target_part, use_llm_keywords
        )
        
        # Generate and display report
        print("Generating detailed report...")
        report = analyzer.create_detailed_report(analysis_results)
        
        print("\n" + report)
        
        # Export results
        filename = f'analysis_{product_type.replace(" ", "_")}'
        if target_part:
            filename += f'_{target_part.replace(" ", "_")}'
        filename += '.json'
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\nAnalysis complete! Results exported to {filename}")
        print(f"Total cost: ${analysis_results['analysis_metadata']['token_usage']['estimated_cost_usd']:.4f}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"Analysis failed: {e}")

    

# Add this at the end of your file
if __name__ == "__main__":
    import sys
    
    app.run(debug=True, port=5000)
