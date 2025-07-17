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
CORS(app) # Enable CORS for all routes, allowing React frontend to connect

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

    def _update_token_usage(self, usage):
        """Update the token usage and estimated cost."""
        if not usage: return
        total_tokens = usage.total_tokens
        cost = total_tokens * (0.375 / 1_000_000) # Approx. cost for gpt-4o-mini
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
                    # The model returns 'Positive', 'Negative', 'Neutral'. We will standardize to uppercase.
                    sentiments.append({'review_index': i, 'sentiment': result['label'].upper(), 'confidence': result['score']})
                except Exception as e:
                    logger.warning(f"Sentiment analysis failed for review {i}: {e}")
                    sentiments.append({'review_index': i, 'sentiment': 'NEUTRAL', 'confidence': 0.5})
        return sentiments

    def extract_issues_with_llm(self, reviews: List[Dict], sentiments: List[Dict], product_type: str) -> List[Dict]:
        """Use OpenAI's GPT to extract specific, actionable issues from negative reviews."""
        logger.info("🤖 Extracting issues from negative reviews with LLM...")
        
        # Create a map of review index to sentiment label
        sentiment_map = {s['review_index']: s['sentiment'] for s in sentiments}

        # Broader filter for negative reviews: rating <= 2 OR explicitly negative sentiment
        negative_reviews = []
        for i, r in enumerate(reviews):
            rating = r.get('review_rating')
            # Check rating first
            if rating is not None and rating <= 2:
                negative_reviews.append(r)
            # If rating is 3 or not present, check sentiment
            elif sentiment_map.get(i) == 'NEGATIVE':
                negative_reviews.append(r)

        if not negative_reviews:
            logger.info(f"No significant negative reviews found for {product_type} to extract issues from.")
            return []
        
        negative_reviews.sort(key=lambda r: len(r.get('review_text', '')), reverse=True)
        sample_reviews = negative_reviews[:20]
        review_texts = [f"Review (Rating: {r.get('review_rating', 'N/A')}): {r.get('review_text', '')[:500]}" for r in sample_reviews]
        combined_text = "\n\n".join(review_texts)
        
        prompt = f"""
        Analyze these negative reviews for a "{product_type}" and extract the top 5 most critical product issues.
        For each issue, provide: issue_name (e.g., "battery_life_poor"), description, frequency, severity ("High", "Medium", "Low"), and a direct example_quote.
        Return a valid JSON object with a single key "issues" containing an array of issue objects.
        Reviews: {combined_text}
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1, max_tokens=2000
            )
            self._update_token_usage(response.usage)
            issues = json.loads(response.choices[0].message.content).get('issues', [])
            for issue in issues:
                issue['type'] = self.categorize_issue(issue.get('issue_name', ''))
            return issues
        except Exception as e:
            logger.error(f"LLM issue extraction failed: {e}")
            return []

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

    def analyze_themes_with_llm(self, reviews: List[Dict], product_type: str) -> Dict[str, Any]:
        """Analyze overarching themes and their sentiment using a smart sample of reviews."""
        logger.info("🎯 Analyzing themes with LLM...")
        if not reviews: return {'themes': {}, 'discovered_themes': [], 'sample_size': 0}

        sample_reviews = self._get_representative_sample(reviews, 15)
        sample_texts = [f"Review (Rating: {r.get('review_rating', 'N/A')}): {r.get('review_text', '')[:400]}" for r in sample_reviews]
        combined_text = "\n\n".join(sample_texts)
        
        prompt = f"""
        Analyze reviews for a "{product_type}". Identify 8-10 main themes. For each theme, provide sentiment (positive/negative/neutral), mention count, key_phrases (array of 2-3 strings), and a representative example_quote.
        Return a single JSON object with a key "themes", containing an object where each key is a theme.
        Example: {{"themes": {{"button_responsiveness": {{"mentions": 5, "sentiment": "positive", "key_phrases": ["clicky buttons", "no input lag"], "example_quote": "..."}}}}}}
        Reviews: {combined_text}
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2, max_tokens=2000
            )
            self._update_token_usage(response.usage)
            theme_analysis = json.loads(response.choices[0].message.content).get('themes', {})
            return {'themes': theme_analysis, 'discovered_themes': list(theme_analysis.keys()), 'sample_size': len(sample_reviews)}
        except Exception as e:
            logger.error(f"LLM theme analysis failed: {e}")
            return {'themes': {}, 'discovered_themes': [], 'sample_size': 0}

    def _get_representative_sample(self, reviews: List[Dict], target_count: int) -> List[Dict]:
        """Create a diverse sample of reviews based on rating and length."""
        if len(reviews) <= target_count: return reviews
        reviews.sort(key=lambda r: len(r.get('review_text', '')), reverse=True)
        high = [r for r in reviews if r.get('review_rating', 3) >= 4]
        low = [r for r in reviews if r.get('review_rating', 3) <= 2]
        mid = [r for r in reviews if r.get('review_rating', 3) == 3]
        sample, seen = [], set()
        for pool in [high, low, mid]:
            for review in pool:
                if len(sample) < target_count and id(review) not in seen:
                    sample.append(review)
                    seen.add(id(review))
        return sample

    def calculate_enhanced_metrics(self, reviews: List[Dict], sentiments: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive metrics from the review data."""
        if not reviews: return {}
        ratings = [r.get('review_rating') for r in reviews if r.get('review_rating') is not None]
        return {
            'total_reviews': len(reviews),
            'average_rating': round(sum(ratings) / len(ratings), 2) if ratings else 0,
            'rating_distribution': dict(Counter(ratings)),
            'sentiment_distribution': dict(Counter(s['sentiment'] for s in sentiments)),
        }

    def generate_comprehensive_analysis(self, reviews: List[Dict], product_type: str) -> Dict[str, Any]:
        """Orchestrate the full analysis pipeline for a single product."""
        logger.info(f"🚀 Starting analysis for '{product_type}'...")
        start_time = time.time()
        sentiments = self.batch_sentiment_analysis(reviews)
        # Pass sentiments to the issue extractor
        issues = self.extract_issues_with_llm(reviews, sentiments, product_type)
        themes = self.analyze_themes_with_llm(reviews, product_type)
        metrics = self.calculate_enhanced_metrics(reviews, sentiments)
        analysis_time = time.time() - start_time
        logger.info(f"✅ Analysis for '{product_type}' completed in {analysis_time:.2f}s.")
        return {
            'product_type': product_type, 'themes': themes, 'issues': issues, 'metrics': metrics,
            'analysis_metadata': {'analysis_date': datetime.now().isoformat(), 'model_used': 'gpt-4o-mini + local RoBERTa', 'analysis_time_seconds': round(analysis_time, 2)},
            'reviews': reviews # Include the raw parsed reviews for frontend use
        }

    def compare_products(self, analysis_a: Dict, analysis_b: Dict) -> Dict[str, Any]:
        """Compare two product analyses and generate strategic insights."""
        name_a = analysis_a.get('product_type', 'Product A')
        name_b = analysis_b.get('product_type', 'Product B')
        logger.info(f"🔍 Comparing '{name_a}' vs '{name_b}'...")

        themes_a = set(analysis_a.get('themes', {}).get('themes', {}).keys())
        themes_b = set(analysis_b.get('themes', {}).get('themes', {}).keys())
        shared_themes = list(themes_a.intersection(themes_b))
        
        comparison_context = f"""
        **Product A ({name_a}):** Avg Rating: {analysis_a['metrics'].get('average_rating', 'N/A')}, Top Issues: {[issue['issue_name'] for issue in analysis_a.get('issues', [])[:3]]}
        **Product B ({name_b}):** Avg Rating: {analysis_b['metrics'].get('average_rating', 'N/A')}, Top Issues: {[issue['issue_name'] for issue in analysis_b.get('issues', [])[:3]]}
        **Shared Themes:** {shared_themes}
        """

        prompt = f"""
        As a product strategy analyst, analyze the data for {name_a} vs {name_b}.
        Data: {comparison_context}
        Generate a JSON object with two keys:
        1. "summary_table": A markdown table comparing sentiment for each shared theme. Columns: Theme, {name_a} Sentiment, {name_b} Sentiment, Winner.
        2. "strategic_insights": An object with keys: "competitive_advantages" (array of strings for {name_a}), "areas_to_improve" (array of strings for {name_a}), and "recommendations" (array of objects with "recommendation", "priority", and "impact").
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3, max_tokens=2000
            )
            self._update_token_usage(response.usage)
            comparison_insights = json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"LLM comparison generation failed: {e}")
            comparison_insights = {"summary_table": "Error", "strategic_insights": {}}
        
        return {
            'shared_themes': shared_themes,
            'unique_to_product_a': list(themes_a - themes_b),
            'unique_to_product_b': list(themes_b - themes_a),
            'theme_sentiment_comparison': self._create_theme_sentiment_comparison(analysis_a, analysis_b, shared_themes),
            'summary_table': comparison_insights.get('summary_table', ''),
            'strategic_insights': comparison_insights.get('strategic_insights', {})
        }

    def _create_theme_sentiment_comparison(self, analysis_a, analysis_b, shared_themes):
        """Helper to create a structured dict of theme sentiment comparison."""
        comparison = {}
        themes_a = analysis_a.get('themes', {}).get('themes', {})
        themes_b = analysis_b.get('themes', {}).get('themes', {})
        for theme in shared_themes:
            comparison[theme] = {
                "product_a": themes_a.get(theme, {}).get('sentiment', 'neutral'),
                "product_b": themes_b.get(theme, {}).get('sentiment', 'neutral'),
            }
        return comparison

# --- HTML Parsing Function ---

def analyze_product_reviews(html_str: str, product_name: str) -> List[Dict]:
    """Parse reviews from HTML string with enhanced error handling."""
    logger.info(f"Parsing HTML for '{product_name}' reviews...")
    try:
        soup = BeautifulSoup(html_str, 'html.parser')
        reviews = []
        
        # Try multiple selectors for the main review container
        review_container_selectors = ['[data-hook="review"]', 'li.review-item', '.review-item', '.review', '[data-testid="review"]']
        
        review_elements = []
        for selector in review_container_selectors:
            elements = soup.select(selector)
            if elements:
                review_elements = elements
                logger.info(f"Found {len(review_elements)} review elements using selector '{selector}'")
                break
        
        if not review_elements:
            logger.warning(f"No review elements found for '{product_name}'. Check HTML or selectors.")
            return []
        
        for i, element in enumerate(review_elements):
            review_data = {'review_id': f"{product_name.replace(' ', '_')}_{i}", 'review_text': '', 'review_rating': None}
            
            # Extract text using multiple possible selectors
            text_selectors = ['[data-hook="review-body"]', 'p.pre-white-space', '.review-text', 'p', '.content']
            text_element = None
            for selector in text_selectors:
                text_element = element.select_one(selector)
                if text_element:
                    review_data['review_text'] = text_element.get_text(strip=True)
                    break
            
            # Extract rating using multiple possible selectors and patterns
            rating_selectors = ['[data-hook="review-star-rating"]', 'p.visually-hidden', '.review-rating']
            rating_element = None
            for selector in rating_selectors:
                rating_element = element.select_one(selector)
                if rating_element:
                    rating_text = rating_element.get_text()
                    # Pattern for "4.0 out of 5 stars"
                    match = re.search(r'(\d+(\.\d)?)\s*out\s*of\s*5', rating_text)
                    if match:
                        review_data['review_rating'] = float(match.group(1))
                        break
                    # Pattern for "Rated 4.5"
                    match = re.search(r'Rated (\d(?:\.\d)?)', rating_text)
                    if match:
                        review_data['review_rating'] = float(match.group(1))
                        break
            
            # Fallback for class-based ratings like 'a-star-4' if no text match
            if review_data['review_rating'] is None and rating_element:
                class_match = re.search(r'a-star-(\d)', ' '.join(rating_element.get('class', [])))
                if class_match:
                    review_data['review_rating'] = float(class_match.group(1))

            if review_data['review_text']:
                reviews.append(review_data)
        
        logger.info(f"Successfully parsed {len(reviews)} reviews for '{product_name}'.")
        return reviews
    
    except Exception as e:
        logger.error(f"An error occurred during HTML parsing for '{product_name}': {e}", exc_info=True)
        return []


# --- Flask API Endpoints ---

@app.route('/api/compare', methods=['POST'])
def handle_compare_request():
    """API endpoint to handle the comparison of two products from their HTML."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return jsonify({"error": "OpenAI API key not configured on the server."}), 500

    try:
        data = request.get_json()
        html_a = data.get('html_a')
        name_a = data.get('product_name_a', 'Product A')
        html_b = data.get('html_b')
        name_b = data.get('product_name_b', 'Product B')

        if not all([html_a, name_a, html_b, name_b]):
            return jsonify({"error": "Missing required fields: html_a, product_name_a, html_b, product_name_b"}), 400

        analyzer = OpenAIHybridReviewAnalyzer(openai_api_key=api_key)

        # Process both products
        reviews_a = analyze_product_reviews(html_a, name_a)
        analysis_a = analyzer.generate_comprehensive_analysis(reviews_a, name_a)

        reviews_b = analyze_product_reviews(html_b, name_b)
        analysis_b = analyzer.generate_comprehensive_analysis(reviews_b, name_b)

        # Compare results
        comparison_result = analyzer.compare_products(analysis_a, analysis_b)

        # Assemble final output
        final_output = {
            "product_a": analysis_a,
            "product_b": analysis_b,
            "comparison": comparison_result
        }
        
        return jsonify(final_output)

    except Exception as e:
        logger.error(f"An error occurred in /api/compare: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

if __name__ == "__main__":
    # Runs the Flask app
    # Use a different port to avoid conflicts with other services
    app.run(debug=True, port=5001)
