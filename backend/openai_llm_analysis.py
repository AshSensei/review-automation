import gc
import json
import re
import os
from collections import Counter
from bs4 import BeautifulSoup
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import time
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIReviewAnalyzer:
    def __init__(self, openai_api_key: str):
        """Initialize the OpenAI-only analyzer."""
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        logger.info("Initialized OpenAI-only analyzer (memory optimized)")
        
        # Token usage tracking
        self.token_usage = {
            'total_tokens': 0,
            'estimated_cost': 0.0
        }
    
    def batch_sentiment_analysis(self, reviews: List[Dict]) -> List[Dict]:
        """Perform sentiment analysis using OpenAI with memory optimization."""
        print("📊 Running OpenAI sentiment analysis...")
        
        # Smaller batches and limit total reviews processed
        batch_size = 8
        max_reviews = min(len(reviews), 100)  # Limit to 100 reviews max
        reviews = reviews[:max_reviews]
        
        sentiments = []
        
        for i in range(0, len(reviews), batch_size):
            batch = reviews[i:i + batch_size]
            
            # Create prompt without storing intermediate strings
            review_texts = []
            for idx, review in enumerate(batch):
                text = review['review_text'][:200]  # Smaller truncation
                rating = review.get('review_rating', 'N/A')
                review_texts.append(f"{i+idx}|{rating}|{text}")
            
            prompt = f"""Analyze sentiment for these reviews (format: index|rating|text):

{chr(10).join(review_texts)}

Return JSON array: [{{"review_index": 0, "sentiment": "positive", "confidence": 0.85}}]"""

            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=500
                )
                
                self.token_usage['total_tokens'] += response.usage.total_tokens
                self.token_usage['estimated_cost'] += response.usage.total_tokens * 0.00015 / 1000
                
                # Parse response and clean up immediately
                response_content = response.choices[0].message.content.strip()
                if response_content.startswith("```json"):
                    response_content = response_content[7:-3].strip()
                
                batch_sentiments = json.loads(response_content)
                sentiments.extend(batch_sentiments)
                
                # Clean up variables
                del response_content, batch_sentiments
                
            except Exception as e:
                logger.warning(f"Sentiment analysis failed for batch {i//batch_size}: {e}")
                # Simple fallback without storing large objects
                for idx, review in enumerate(batch):
                    rating = review.get('review_rating', 3)
                    if rating >= 4:
                        sentiment, confidence = 'positive', 0.7
                    elif rating <= 2:
                        sentiment, confidence = 'negative', 0.7
                    else:
                        sentiment, confidence = 'neutral', 0.5
                    
                    sentiments.append({
                        'review_index': i + idx,
                        'sentiment': sentiment,
                        'confidence': confidence
                    })
            
            # Clean up batch data
            del review_texts, batch
            
            # Force garbage collection every few batches
            if i % (batch_size * 3) == 0:
                gc.collect()
        
        return sentiments
    
    def extract_issues_with_llm(self, reviews: List[Dict], product_type: str) -> List[Dict]:
        """Extract specific issues using OpenAI with memory optimization."""
        print("🤖 Extracting issues with LLM...")
        
        # Filter and limit negative reviews more aggressively
        negative_reviews = []
        for review in reviews[:50]:  # Only check first 50 reviews
            rating = review.get('review_rating', 3)
            if rating <= 2:
                negative_reviews.append(review)
            if len(negative_reviews) >= 10:  # Max 10 negative reviews
                break
        
        if not negative_reviews:
            # Take first 5 reviews if no negatives found
            negative_reviews = reviews[:5]
        
        # Create compact prompt without storing large intermediate strings
        texts = [f"{i}: {r['review_text'][:250]}" for i, r in enumerate(negative_reviews)]
        
        prompt = f"""Find issues in these {product_type} reviews:

{chr(10).join(texts)}

Return JSON: [{{"issue_name": "name", "description": "desc", "frequency": 1, "severity": "high", "example_quote": "quote"}}]"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800
            )
            
            self.token_usage['total_tokens'] += response.usage.total_tokens
            self.token_usage['estimated_cost'] += response.usage.total_tokens * 0.00015 / 1000

            response_content = response.choices[0].message.content.strip()
            if response_content.startswith("```json"):
                response_content = response_content[7:-3].strip()

            issues = json.loads(response_content)
            
            # Add metadata without creating new objects
            for issue in issues:
                issue['type'] = self.categorize_issue(issue['issue_name'])
                issue['review_count'] = len(negative_reviews)
            
            # Clean up
            del response_content, texts
            return issues[:5]  # Limit to 5 issues
            
        except Exception as e:
            logger.error(f"LLM issue extraction failed: {e}")
            return []
    
    def categorize_issue(self, issue_name: str) -> str:
        """Categorize issues into types."""
        issue_name = issue_name.lower()
        
        categories = {
            'connectivity': ['connection', 'disconnect', 'bluetooth', 'wifi', 'pair'],
            'build_quality': ['build', 'quality', 'cheap', 'flimsy', 'material'],
            'performance': ['slow', 'lag', 'performance', 'speed', 'responsive'],
            'battery': ['battery', 'charge', 'power', 'drain'],
            'durability': ['break', 'broke', 'broken', 'durable', 'last'],
            'comfort': ['comfort', 'hurt', 'pain', 'ergonomic'],
            'functionality': ['work', 'function', 'feature', 'button']
        }
        
        for category, keywords in categories.items():
            if any(keyword in issue_name for keyword in keywords):
                return category
        
        return 'other'
    
    def analyze_themes_with_llm(self, reviews: List[Dict], product_type: str) -> Dict[str, Any]:
        """Analyze themes using OpenAI with memory optimization."""
        print("🎯 Analyzing themes with LLM...")
        
        # Very small sample to minimize memory usage
        sample_reviews = self.get_representative_sample(reviews, 6)
        
        # Create compact prompt
        texts = [f"{i}: {r['review_text'][:200]}" for i, r in enumerate(sample_reviews)]
        
        prompt = f"""Analyze these {product_type} reviews. Return JSON:

{chr(10).join(texts)}

{{"discovered_themes": ["theme1", "theme2"], "themes": {{"theme1": {{"mentions": 2, "sentiment": "positive", "confidence": 0.8, "key_phrases": ["good"], "example_quote": "works well"}}}}}}"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800
            )
            
            self.token_usage['total_tokens'] += response.usage.total_tokens
            self.token_usage['estimated_cost'] += response.usage.total_tokens * 0.00015 / 1000

            response_content = response.choices[0].message.content.strip()
            if response_content.startswith("```json"):
                response_content = response_content[7:-3].strip()
                
            theme_analysis = json.loads(response_content)
            theme_analysis['sample_size'] = len(sample_reviews)
            
            # Clean up
            del response_content, texts
            return theme_analysis
            
        except Exception as e:
            logger.error(f"LLM theme analysis failed: {e}")
            return {'themes': {}, 'discovered_themes': [], 'sample_size': 0}
    
    def get_representative_sample(self, reviews: List[Dict], target_count: int) -> List[Dict]:
        """Get diverse representative sample of reviews."""
        
        if len(reviews) <= target_count:
            return reviews
        
        # Categorize reviews by rating if available
        high_rated = [r for r in reviews if r.get('review_rating', 3) >= 4]
        low_rated = [r for r in reviews if r.get('review_rating', 3) <= 2]
        mid_rated = [r for r in reviews if r.get('review_rating', 3) == 3]
        
        # Distribute sample across rating categories
        sample = []
        
        # Get mix of ratings
        sample.extend(high_rated[:target_count//3])
        sample.extend(low_rated[:target_count//3])
        sample.extend(mid_rated[:target_count//3])
        
        # Fill remaining with any reviews
        remaining = target_count - len(sample)
        if remaining > 0:
            other_reviews = [r for r in reviews if r not in sample]
            sample.extend(other_reviews[:remaining])
        
        return sample
    
    def generate_insights_with_llm(self, themes: Dict, issues: List[Dict], metrics: Dict, product_type: str) -> Dict:
        """Generate executive summary and recommendations using LLM."""
        print("💡 Generating insights with LLM...")
        
        # Prepare context for LLM
        context = f"""Product Analysis Context:
Product Type: {product_type}
Total Reviews: {metrics['total_reviews']}
Average Rating: {metrics['average_rating']:.1f}/5
Sentiment Distribution: {metrics['sentiment_distribution']}

Top Themes:
{json.dumps(themes.get('themes', {}), indent=2)}

Key Issues:
{json.dumps(issues[:5], indent=2)}"""

        prompt = f"""Based on this product analysis, provide:

1. Executive summary (2-3 sentences highlighting key findings)
2. Top 5 actionable recommendations for improvement
3. Priority level for each recommendation (high/medium/low)
4. Business impact assessment

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
            
            self.token_usage['total_tokens'] += response.usage.total_tokens
            self.token_usage['estimated_cost'] += response.usage.total_tokens * 0.00015 / 1000
        
            # Get the response content and strip markdown
            response_content = response.choices[0].message.content.strip()
            if response_content.startswith("```json"):
                response_content = response_content[7:-3].strip()

            return json.loads(response_content)
            
        except Exception as e:
            logger.error(f"LLM insights generation failed: {e}")
            return {
                "executive_summary": "Analysis completed with mixed results requiring attention.",
                "recommendations": [{"recommendation": "Review customer feedback patterns", "priority": "medium", "impact": "moderate"}],
                "key_insights": ["Customer feedback analysis completed"]
            }
    
    def calculate_enhanced_metrics(self, reviews: List[Dict], sentiments: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive metrics."""
        
        ratings = [r.get('review_rating', 3) for r in reviews]
        review_lengths = [len(r['review_text']) for r in reviews if r['review_text'].strip()]
        
        # Sentiment distribution
        sentiment_counts = Counter(s['sentiment'] for s in sentiments)
        
        # Confidence scores
        confidence_scores = [s['confidence'] for s in sentiments]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # Simple reading ease approximation (without textstat)
        avg_reading_ease = 0
        if review_lengths:
            avg_length = sum(review_lengths) / len(review_lengths)
            # Simple approximation: shorter reviews = easier to read
            avg_reading_ease = max(0, 100 - (avg_length / 10))
        
        return {
            'total_reviews': len(reviews),
            'average_rating': sum(ratings) / len(ratings) if ratings else 0,
            'rating_distribution': dict(Counter(ratings)),
            'average_review_length': sum(review_lengths) / len(review_lengths) if review_lengths else 0,
            'sentiment_distribution': dict(sentiment_counts),
            'reading_ease_score': avg_reading_ease,
            'average_sentiment_confidence': avg_confidence,
            'analysis_quality': 'enhanced' if avg_confidence > 0.7 else 'standard'
        }
    
    def fallback_issue_extraction(self, reviews: List[Dict]) -> List[Dict]:
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
                    
                    issues.append({
                        'issue_name': issue_description,
                        'description': f"Pattern match: {issue_description}",
                        'frequency': 1,
                        'severity': 'medium',
                        'example_quote': text[:100] + "...",
                        'type': issue_type
                    })
        
        # Aggregate duplicates
        issue_counts = Counter(issue['issue_name'] for issue in issues)
        unique_issues = []
        for issue in issues:
            if issue['issue_name'] not in [ui['issue_name'] for ui in unique_issues]:
                issue['frequency'] = issue_counts[issue['issue_name']]
                unique_issues.append(issue)
        
        return sorted(unique_issues, key=lambda x: x['frequency'], reverse=True)[:5]
    
    def fallback_theme_analysis(self, reviews: List[Dict], product_type: str) -> Dict[str, Any]:
        """Fallback theme analysis using keyword matching."""
        logger.info("Using fallback theme analysis")
        
        # Default themes by product type
        default_themes = {
            'gaming controller': ['build_quality', 'battery_life', 'connectivity', 'comfort', 'performance'],
            'headphones': ['sound_quality', 'comfort', 'build_quality', 'battery_life', 'connectivity'],
            'default': ['quality', 'performance', 'design', 'price', 'durability']
        }
        
        themes = default_themes.get(product_type.lower(), default_themes['default'])
        
        theme_analysis = {}
        for theme in themes:
            theme_analysis[theme] = {
                'mentions': 0,
                'sentiment': 'neutral',
                'confidence': 0.5,
                'key_phrases': [],
                'example_quote': ''
            }
        
        return {
            'themes': theme_analysis,
            'discovered_themes': themes,
            'sample_size': len(reviews)
        }
    
    def generate_comprehensive_analysis(self, reviews: List[Dict], product_type: str = "product") -> Dict[str, Any]:
        """Generate comprehensive analysis using OpenAI only."""
        
        print(f"🚀 Starting OpenAI-only analysis for {len(reviews)} reviews...")
        start_time = time.time()
        
        # Force garbage collection
        gc.collect()
        
        # Step 1: OpenAI sentiment analysis
        sentiments = self.batch_sentiment_analysis(reviews)
        
        # Step 2: LLM-powered issue extraction
        issues = self.extract_issues_with_llm(reviews, product_type)
        
        # Step 3: LLM-powered theme analysis
        themes = self.analyze_themes_with_llm(reviews, product_type)
        
        # Step 4: Enhanced metrics calculation
        metrics = self.calculate_enhanced_metrics(reviews, sentiments)
        
        # Step 5: LLM-powered insights generation
        insights = self.generate_insights_with_llm(themes, issues, metrics, product_type)
        
        analysis_time = time.time() - start_time
        
        # Final garbage collection
        gc.collect()
        
        return {
            'themes': themes,
            'issues': issues,
            'metrics': metrics,
            'insights': insights,
            'analysis_metadata': {
                'total_reviews': len(reviews),
                'analysis_date': datetime.now().isoformat(),
                'product_type': product_type,
                'model_used': 'gpt-4o-mini (OpenAI only)',
                'analysis_time_seconds': round(analysis_time, 2),
                'token_usage': self.token_usage.copy()
            }
        }

# HTML parsing function
def parse_reviews_from_html_string(html_str: str) -> Optional[List[Dict]]:
    """Parse reviews from HTML string with memory optimization."""
    try:
        # Limit HTML size to prevent memory issues
        max_html_size = 2 * 1024 * 1024  # 2MB limit
        if len(html_str) > max_html_size:
            logger.warning(f"HTML too large ({len(html_str)} bytes), truncating to {max_html_size}")
            html_str = html_str[:max_html_size]
        
        # Use lxml parser if available (more memory efficient), fallback to html.parser
        try:
            soup = BeautifulSoup(html_str, 'lxml')
        except:
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
            # Clean up immediately
            del soup
            gc.collect()
            return []
        
        # Process elements and clean up as we go
        for element in review_elements:
            review_data = {'review_text': '', 'review_rating': None}
            
            # Extract text
            text_selectors = ['p.pre-white-space', '.review-text', 'p', '.content']
            for selector in text_selectors:
                text_element = element.select_one(selector)
                if text_element:
                    text = text_element.get_text(strip=True)
                    # Limit individual review length
                    review_data['review_text'] = text[:2000] if len(text) > 2000 else text
                    break
            
            # Extract rating
            rating_element = element.select_one('p.visually-hidden')
            if rating_element:
                rating_match = re.search(r'Rated (\d(?:\.\d)?)', rating_element.text)
                if rating_match:
                    review_data['review_rating'] = float(rating_match.group(1))
            
            if review_data['review_text']:
                reviews.append(review_data)
                
            # Limit total reviews to prevent memory issues
            if len(reviews) >= 200:
                logger.warning("Reached review limit (200), stopping parsing")
                break
        
        # Clean up soup object immediately
        del soup
        gc.collect()
        
        logger.info(f"Successfully parsed {len(reviews)} reviews")
        return reviews
    
    except Exception as e:
        logger.error(f"Error parsing HTML: {e}")
        # Ensure cleanup on error
        gc.collect()
        return None

# Flask app
app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:3000",
    "http://localhost:5173",  # Vite dev server
    "https://*.vercel.app",   # All Vercel subdomains
    r"https://.*\.vercel\.app"  # Regex pattern for Vercel
])

@app.route('/api/analyze-html', methods=['POST'])
def analyze_html():
    try:
        # Aggressive memory management
        gc.collect()
        
        # Get HTML from request
        data = request.get_json()
        html_content = data.get('html', '')
        
        if not html_content:
            return jsonify({'error': 'No HTML content provided'}), 400
        
        # Check HTML size before processing
        if len(html_content) > 3 * 1024 * 1024:  # 3MB limit
            return jsonify({'error': 'HTML file too large (max 3MB)'}), 400
        
        # Parse reviews from HTML string
        reviews = parse_reviews_from_html_string(html_content)
        
        # Clean up HTML content immediately
        del html_content, data
        gc.collect()
        
        if not reviews:
            return jsonify({'error': 'No reviews found in HTML'}), 400
        
        # Limit reviews for memory safety
        if len(reviews) > 50:
            reviews = reviews[:50]
            logger.warning(f"Limited to first 50 reviews for memory safety")
        
        # Get product type (or use default)
        product_type = request.get_json().get('product_type', 'product') if request.get_json() else 'product'
        
        # Initialize analyzer
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            return jsonify({'error': 'OpenAI API key not configured'}), 500
        
        analyzer = OpenAIReviewAnalyzer(openai_api_key)
        
        # Generate analysis
        analysis_results = analyzer.generate_comprehensive_analysis(reviews, product_type)
        
        # Clean up analyzer
        del analyzer
        gc.collect()
        
        # Create minimal response to save memory
        response_data = {
            'reviews': [r['review_text'][:500] for r in reviews[:20]],  # Limit and truncate
            'sentiment': analysis_results['metrics']['sentiment_distribution'],
            'themes': analysis_results['themes'],
            'issues': analysis_results['issues'][:3],  # Limit issues
            'insights': analysis_results['insights'],
            'summary': analysis_results['insights'].get('executive_summary', ''),
            'analysis_metadata': {
                'total_reviews': analysis_results['analysis_metadata']['total_reviews'],
                'analysis_time_seconds': analysis_results['analysis_metadata']['analysis_time_seconds'],
                'estimated_cost': analysis_results['analysis_metadata']['token_usage']['estimated_cost']
            }
        }
        
        # Clean up analysis results
        del analysis_results, reviews
        gc.collect()
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"API error: {e}")
        gc.collect()  # Clean up on error
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)