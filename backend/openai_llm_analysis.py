import gc
import json
import re
import os
from collections import Counter
from bs4 import BeautifulSoup
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import nltk
from textstat import flesch_reading_ease
import time
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

load_dotenv()

# Memory-safety caps
MAX_HTML_SIZE = 3 * 1024 * 1024  # 3MB
PARSE_HTML_LIMIT = 30  # max reviews to parse
ANALYSIS_REVIEW_LIMIT = 30  # max reviews to analyze
NEGATIVE_SAMPLE = 5  # negative reviews sample size

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_reviews_comprehensive(html_content: str) -> List[Dict[str, Any]]:
    """
    Comprehensive HTML parsing that combines both approaches.
    Uses your existing logic plus fallback methods from the original code.
    """
    try:
        # Check HTML size limit
        if len(html_content) > MAX_HTML_SIZE:
            logger.warning(f"HTML content too large: {len(html_content)} bytes, truncating to {MAX_HTML_SIZE}")
            html_content = html_content[:MAX_HTML_SIZE]

        soup = BeautifulSoup(html_content, 'html.parser')
        reviews = []

        # Your existing selectors (prioritized for your specific use case)
        primary_selectors = ['li.review-item', '.review-item', '.review', '[data-testid="review"]']

        # Additional fallback selectors from original code
        fallback_selectors = [
            # Amazon-style reviews
            '[data-hook="review"]',
            # Generic review patterns
            '.review-container', '.user-review', '.customer-review',
            # Google/Yelp style
            '[data-review-id]',
            # Broad fallback
            'div, article, section'
        ]

        all_selectors = primary_selectors + fallback_selectors

        review_elements = []
        successful_selector = None

        for selector in all_selectors:
            elements = soup.select(selector)
            if elements:
                review_elements = elements[:PARSE_HTML_LIMIT]  # Apply limit
                successful_selector = selector
                logger.info(f"Found {len(elements)} review elements using selector: {selector}")
                break

        if not review_elements:
            logger.warning("No review elements found with any selector")
            return []

        for i, element in enumerate(review_elements):
            try:
                review_data = {'review_text': '', 'review_rating': None, 'review_id': f'review_{i}'}

                # Extract review text using multiple strategies
                review_text = ""

                # Your existing text extraction logic
                text_selectors = ['p.pre-white-space', '.review-text', 'p', '.content']
                for selector in text_selectors:
                    text_element = element.select_one(selector)
                    if text_element:
                        review_text = text_element.get_text(strip=True)
                        break

                # Fallback text extraction from original code
                if not review_text:
                    # Try Amazon-style selectors
                    amazon_text = element.select('[data-hook="review-body"] span')
                    if amazon_text:
                        review_text = ' '.join([elem.get_text(strip=True) for elem in amazon_text])
                    else:
                        # Generic fallback
                        review_text = element.get_text(strip=True)

                # Skip if no meaningful text
                if not review_text or len(review_text) < 10:
                    continue

                review_data['review_text'] = review_text[:1000]  # Limit length

                # Extract rating using multiple strategies
                rating = None

                # Your existing rating extraction
                rating_element = element.select_one('p.visually-hidden')
                if rating_element:
                    rating_match = re.search(r'Rated (\d(?:\.\d)?)', rating_element.text)
                    if rating_match:
                        rating = float(rating_match.group(1))

                # Fallback rating extraction from original code
                if rating is None:
                    rating_selectors = [
                        '[data-hook="review-star-rating"]',
                        '.rating', '.stars', '.review-rating',
                        '[aria-label*="star"]'
                    ]

                    for rating_selector in rating_selectors:
                        rating_elements = element.select(rating_selector)
                        for rating_elem in rating_elements:
                            rating_text = rating_elem.get_text(strip=True)

                            # Look for numeric ratings
                            rating_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:out of|\/|\s)\s*(\d+)', rating_text)
                            if rating_match:
                                rating = float(rating_match.group(1))
                                break

                            # Look for star counts
                            star_match = re.search(r'(\d+(?:\.\d+)?)\s*star', rating_text.lower())
                            if star_match:
                                rating = float(star_match.group(1))
                                break

                            # Check aria-label
                            aria_label = rating_elem.get('aria-label', '')
                            if aria_label:
                                aria_rating = re.search(r'(\d+(?:\.\d+)?)\s*(?:out of|star)', aria_label.lower())
                                if aria_rating:
                                    rating = float(aria_rating.group(1))
                                    break

                        if rating is not None:
                            break

                review_data['review_rating'] = rating
                review_data['source_element'] = successful_selector

                reviews.append(review_data)

            except Exception as e:
                logger.debug(f"Failed to parse individual review {i}: {e}")
                continue

        # If no structured reviews found, try text block extraction
        if not reviews:
            logger.info("No structured reviews found, attempting text block extraction")
            text_blocks = soup.find_all(text=True)
            potential_reviews = []

            for text in text_blocks:
                text = text.strip()
                if (len(text) > 50 and
                    len(text) < 2000 and
                    not text.startswith('<') and
                    'script' not in text.lower() and
                    'style' not in text.lower()):
                    potential_reviews.append(text)

            # Take the longest text blocks as potential reviews
            potential_reviews.sort(key=len, reverse=True)
            for i, text in enumerate(potential_reviews[:PARSE_HTML_LIMIT]):
                reviews.append({
                    'review_text': text,
                    'review_rating': None,
                    'review_id': f"text_block_{i}",
                    'source_element': 'text_extraction'
                })

        logger.info(f"Successfully parsed {len(reviews)} reviews from HTML")
        return reviews[:PARSE_HTML_LIMIT]

    except Exception as e:
        logger.error(f"Failed to parse HTML: {e}")
        return []

class OpenAIReviewAnalyzer:
    def __init__(self, openai_api_key: str):
        """Initialize the analyzer with OpenAI."""
        self.openai_client = OpenAI(api_key=openai_api_key)

        # Token usage tracking
        self.token_usage = {
            'total_tokens': 0,
            'estimated_cost': 0.0
        }

    def _track_usage(self, usage):
        """Track OpenAI API usage."""
        if usage:
            tokens = usage.total_tokens
            self.token_usage['total_tokens'] += tokens
            # Using gpt-5 pricing: $0.150 / 1M input tokens
            self.token_usage['estimated_cost'] += tokens * 1.25 / 1000000

    def batch_sentiment_analysis(self, reviews: List[Dict]) -> List[Dict]:
        """
        Performs sentiment analysis on a batch of reviews using the OpenAI API.
        """
        logger.info("📊 Running batch sentiment analysis with OpenAI...")
        sentiments = []
        # Filter out empty reviews before sending to API
        valid_reviews = [(i, r) for i, r in enumerate(reviews) if r.get('review_text', '').strip()]
        
        if not valid_reviews:
            return []

        # Create a single prompt for all valid reviews
        prompt_reviews = "\n".join([f'{{"index": {i}, "text": {json.dumps(r["review_text"][:1000])}}}' for i, r in valid_reviews])

        prompt = f"""
Analyze the sentiment for each JSON object in the following list of reviews.
For each review, determine if the sentiment is "positive", "negative", or "neutral".
Respond with a single JSON object with a key "sentiments" which contains a JSON array. Each object in the array must contain the original 'index' and the determined 'sentiment'.
DO NOT include any other text, explanations, or markdown.

Example response format:
{{
  "sentiments": [
    {{"index": 0, "sentiment": "positive"}},
    {{"index": 1, "sentiment": "negative"}}
  ]
}}

Reviews to analyze:
{prompt_reviews}
"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_completion_tokens=2000,
                response_format={"type": "json_object"}
            )
            self._track_usage(response.usage)
            content = response.choices[0].message.content.strip()
            result_data = json.loads(content)
            llm_sentiments = result_data.get('sentiments', [])

            # Create a dictionary for quick lookup
            sentiment_map = {item['index']: item['sentiment'].lower() for item in llm_sentiments if 'index' in item and 'sentiment' in item}

            # Map the results back to the original review list
            for i, review in enumerate(reviews):
                sentiment_label = sentiment_map.get(i, 'neutral') # Default to neutral if API didn't return a value for this index
                sentiments.append({
                    'review_index': i,
                    'sentiment': sentiment_label,
                    'confidence': 0.98,  # High confidence for LLM analysis
                    'review_rating': review.get('review_rating')
                })
        except Exception as e:
            logger.error(f"OpenAI sentiment analysis failed: {e}. Falling back to rating-based sentiment.")
            # Fallback logic if API fails
            for i, review in enumerate(reviews):
                rating = review.get('review_rating')
                sentiment, conf = 'neutral', 0.5
                if rating is not None:
                    if rating >= 4:
                        sentiment, conf = 'positive', 0.7
                    elif rating <= 2:
                        sentiment, conf = 'negative', 0.7
                sentiments.append({
                    'review_index': i,
                    'sentiment': sentiment,
                    'confidence': conf,
                    'review_rating': rating
                })
        return sentiments

    def extract_issues_with_llm(self, reviews: List[Dict], product_type: str) -> List[Dict]:
        """Enhanced issue extraction with better prompting from original code."""
        print("🤖 Extracting issues with enhanced LLM...")

        # Get negative reviews more systematically
        negative_reviews = [r for r in reviews if r.get('review_rating', 3) <= 2]
        # If there are no negative reviews, consider all of them
        if not negative_reviews:
            negative_reviews = reviews
            
        # Enhanced prompt from original code
        prompt = f"""
You are an expert at analyzing user feedback for {product_type} products.

Please follow these steps:
1. Carefully read the customer reviews provided below.
2. Identify recurring or serious issues mentioned by multiple users.
3. Group similar complaints together as one "issue".
4. For each issue:
    - Give it a short but descriptive "issue_name"
    - Summarize it briefly under "description"
    - Rate its severity as "high", "medium", or "low" based on impact
    - Count how many users mention this issue as "frequency"
    - Choose one quote that clearly illustrates the issue as "example_quote"

Return the final result as a JSON array, formatted EXACTLY like this:
[
 {{
  "issue_name": "unique_name",
  "description": "brief description",
  "severity": "high|medium|low",
  "frequency": number,
  "example_quote": "..."
 }},...
]

Do not explain anything outside the JSON.

Reviews:
""" + "\n".join(f"Review {i+1}: {r['review_text'][:250]}" for i, r in enumerate(negative_reviews))

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_completion_tokens=800
            )

            self._track_usage(response.usage)

            # Clean response content
            content = re.sub(r"^```json|```$", "", response.choices[0].message.content.strip())
            issues = json.loads(content)

            # Enrich with additional data
            enriched_issues = []
            for issue in issues[:5]:
                name = issue.get("issue_name", "")
                enriched_issues.append({
                    "issue_name": name,
                    "description": issue.get("description", ""),
                    "severity": issue.get("severity", "medium"),
                    "review_count": issue.get("frequency", len(negative_reviews)),
                    "example_quote": issue.get("example_quote", ""),
                    "related_quotes": [issue.get("example_quote", "")],
                    "type": self.categorize_issue(name)
                })

            return enriched_issues

        except Exception as e:
            logger.error(f"LLM issue extraction failed: {e}")
            return self.fallback_issue_extraction(negative_reviews)

    def categorize_issue(self, issue_name: str) -> str:
        """Enhanced issue categorization combining both approaches."""
        issue_name_lower = issue_name.lower()

        # Your existing categories
        categories = {
            'connectivity': ['connection', 'disconnect', 'bluetooth', 'wifi', 'pair'],
            'build_quality': ['build', 'quality', 'cheap', 'flimsy', 'material'],
            'performance': ['slow', 'lag', 'performance', 'speed', 'responsive'],
            'battery': ['battery', 'charge', 'power', 'drain'],
            'durability': ['break', 'broke', 'broken', 'durable', 'last'],
            'comfort': ['comfort', 'hurt', 'pain', 'ergonomic'],
            'functionality': ['work', 'function', 'feature', 'button']
        }

        # Additional categories from original code
        additional_categories = {
            'logistics': ['shipping', 'delivery', 'arrival', 'package'],
            'pricing': ['price', 'cost', 'expensive', 'value'],
            'customer_service': ['customer', 'service', 'support', 'help'],
            'sizing': ['size', 'fit', 'dimension', 'scale']
        }

        # Merge categories
        all_categories = {**categories, **additional_categories}

        for category, keywords in all_categories.items():
            if any(keyword in issue_name_lower for keyword in keywords):
                return category

        return 'other'

    def analyze_themes_with_llm(self, reviews: List[Dict], product_type: str) -> Dict[str, Any]:
        """Enhanced theme analysis with better sampling."""
        print("🎯 Analyzing themes with enhanced LLM...")

        # no need to sample for small review set
        sample_reviews = reviews  

        # Enhanced prompting
        lines = [
            f"{i}|{r.get('review_rating','N/A')}|{r['review_text'][:200]}"
            for i, r in enumerate(sample_reviews)
        ]

        prompt = f"""
You are a detail-oriented product analyst for {product_type}. Your task is to analyze customer reviews.

Follow this two-step process:
1.  **Analysis Step:** Read all reviews and identify 5-8 major themes. For each theme, create a list of the specific positive points and a separate list of the specific negative points or complaints mentioned by users.
2.  **Formatting Step:** Based on your analysis from Step 1, populate the final JSON structure below. Do not include your reasoning in the final JSON.

The JSON output must have a root key "themes", containing an object for each discovered theme. Each theme object must include sentiment, confidence, and the lists of 'positives' and 'negatives'.

Example Output Structure:
{{
  "themes": {{
    "Durability": {{
      "sentiment": "mixed",
      "confidence": 0.85,
      "positives": ["No stick drift after a year"],
      "negatives": ["Thumbstick plastic grinds on friction rings", "Shoulder buttons failed"]
    }}
  }}
}}

Reviews to Analyze:
""" + "\n".join(lines)

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_completion_tokens=1000
            )

            self._track_usage(response.usage)

            content = re.sub(r"^```json|```$", "", response.choices[0].message.content.strip())
            themes = json.loads(content)

            return themes

        except Exception as e:
            logger.error(f"Theme analysis failed: {e}")
            return self.fallback_theme_analysis(reviews, product_type)

    def generate_insights_with_llm(self, themes: Dict, issues: List[Dict], metrics: Dict, product_type: str) -> Dict:
        """Enhanced insights generation with better context."""
        print("💡 Generating insights with enhanced LLM...")

        context = {"product_type": product_type, "themes": themes, "issues": issues, "metrics": metrics}

        prompt = """
You are a senior product strategist. Based on the following analysis data, please do the following:

Step 1: Identify key patterns, challenges, or opportunities from the data, paying close attention to the specific `positives` and `negatives` listed within each theme.
Step 2: Write a clear and concise 2–3 sentence executive summary. Crucially, use the specific details from your analysis instead of generic terms (e.g., mention 'thumbstick grinding' instead of 'durability concerns').
Step 3: Generate 3 to 5 actionable product or business recommendations that directly address the specific `negatives` you found. For each recommendation, provide the recommendation, priority, impact, and rationale.
Step 4: Summarize 3 to 6 additional key insights or observations based on the specific `positives` and `negatives`.

Return the result using this JSON format ONLY:
{
  "executive_summary": "",
  "recommendations": [
    {
      "recommendation": "",
      "priority": "high|medium|low",
      "impact": "",
      "rationale": ""
    }
  ],
  "key_insights": ["..."]
}

Do not include anything outside this JSON.

Analysis Data:
""" + json.dumps(context)
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_completion_tokens=1000
            )

            self._track_usage(response.usage)

            content = re.sub(r"^```json|```$", "", response.choices[0].message.content.strip())
            insights = json.loads(content)

            return insights

        except Exception as e:
            logger.error(f"Insights generation failed: {e}")
            return {
                "executive_summary": "Analysis completed with mixed results requiring attention.",
                "recommendations": [{"recommendation": "Review customer feedback patterns", "priority": "medium", "impact": "moderate", "rationale": "Based on analysis"}],
                "key_insights": ["Customer feedback analysis completed"]
            }

    def get_representative_sample(self, reviews: List[Dict], target_count: int) -> List[Dict]:
        """Get diverse representative sample of reviews."""

        if len(reviews) <= target_count:
            return reviews

        # Categorize reviews
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

    def calculate_enhanced_metrics(self, reviews: List[Dict], sentiments: List[Dict]) -> Dict[str, Any]:
        """Enhanced metrics calculation combining both approaches."""

        ratings = [r.get('review_rating') for r in reviews if r.get('review_rating') is not None]
        review_lengths = [len(r['review_text']) for r in reviews if r['review_text'].strip()]

        # Sentiment distribution
        sentiment_counts = Counter(s['sentiment'] for s in sentiments)

        # Reading complexity
        avg_reading_ease = 0
        if review_lengths:
            try:
                combined_text = " ".join([r['review_text'] for r in reviews[:10]])
                avg_reading_ease = flesch_reading_ease(combined_text)
            except:
                avg_reading_ease = 0

        # Additional metrics
        confidence_scores = [s['confidence'] for s in sentiments]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

        # Rating stats
        rating_stats = {}
        if ratings:
            rating_stats = {
                'average_rating': round(sum(ratings) / len(ratings), 2),
                'rating_count': len(ratings),
                'rating_distribution': dict(Counter(ratings))
            }

        return {
            'total_reviews': len(reviews),
            'average_rating': sum(ratings) / len(ratings) if ratings else 0,
            'rating_distribution': dict(Counter(ratings)),
            'average_review_length': sum(review_lengths) / len(review_lengths) if review_lengths else 0,
            'sentiment_distribution': dict(sentiment_counts),
            'reading_ease_score': avg_reading_ease,
            'average_sentiment_confidence': avg_confidence,
            'analysis_quality': 'enhanced' if avg_confidence > 0.7 else 'standard',
            'rating_stats': rating_stats,
            'text_stats': {
                'average_length': round(sum(review_lengths) / len(review_lengths), 0) if review_lengths else 0,
                'min_length': min(review_lengths) if review_lengths else 0,
                'max_length': max(review_lengths) if review_lengths else 0
            },
            'analysis_coverage': {
                'sentiment_analyzed': len(sentiments),
                'ratings_available': len(ratings),
                'text_reviews': len([r for r in reviews if r.get('review_text')])
            }
        }

    def fallback_issue_extraction(self, reviews: List[Dict]) -> List[Dict]:
        """Enhanced fallback issue extraction."""
        logger.info("Using enhanced fallback issue extraction")

        issue_patterns = [
            (r"(break|broke|broken|fail|failed|stop|stopped)\s+(\w+)", "failure"),
            (r"(doesn't|don't|won't|can't)\s+(\w+)", "functionality"),
            (r"(poor|bad|terrible|awful|horrible)\s+(\w+)", "quality"),
            (r"(disconnect|connection|connectivity)", "connectivity"),
            (r"(battery|charge|power)\s+(drain|die|dead)", "battery"),
            (r"(uncomfortable|hurt|pain)", "comfort")
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
                        'type': issue_type,
                        'review_count': len(reviews),
                        'related_quotes': [text[:100] + "..."]
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
        """Enhanced fallback theme analysis."""
        logger.info("Using enhanced fallback theme analysis")

        # Default themes by product type
        default_themes = {
            'gaming controller': ['build_quality', 'battery_life', 'connectivity', 'comfort', 'performance'],
            'headphones': ['sound_quality', 'comfort', 'build_quality', 'battery_life', 'connectivity'],
            'product': ['quality', 'performance', 'design', 'price', 'durability']
        }

        themes = default_themes.get(product_type.lower(), default_themes['product'])

        theme_analysis = {}
        for theme in themes:
            theme_analysis[theme] = {
                'mentions': len(reviews) // len(themes),  # Rough estimate
                'sentiment': 'neutral',
                'confidence': 0.5,
                'key_phrases': [theme.replace('_', ' ')],
                'example_quote': f'Analysis of {theme.replace("_", " ")}'
            }

        return {
            'themes': theme_analysis,
            'discovered_themes': themes,
            'sample_size': len(reviews)
        }

    def generate_comprehensive_analysis(self, reviews: List[Dict], product_type: str = "product") -> Dict[str, Any]:
        """Generate comprehensive analysis using the OpenAI API."""

        print(f"🚀 Starting analysis for {len(reviews)} reviews...")
        start_time = time.time()

        # Step 1: Sentiment analysis
        sentiments = self.batch_sentiment_analysis(reviews)

        # Step 2: Issue extraction
        issues = self.extract_issues_with_llm(reviews, product_type)

        # Step 3: Theme analysis
        themes = self.analyze_themes_with_llm(reviews, product_type)

        # Step 4: Metrics calculation
        metrics = self.calculate_enhanced_metrics(reviews, sentiments)

        # Step 5: Insights generation
        insights = self.generate_insights_with_llm(themes, issues, metrics, product_type)

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
                'model_used': 'gpt-5',
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
        lines.append(f"Reviews Analyzed: {metadata['total_reviews']}")
        lines.append(f"Analysis Date: {metadata['analysis_date']}")
        lines.append(f"Model: {metadata['model_used']}")
        lines.append(f"Processing Time: {metadata['analysis_time_seconds']}s")
        lines.append(f"API Cost: ${metadata['token_usage']['estimated_cost']:.4f}")
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

        # Top Issues
        lines.append("TOP ISSUES IDENTIFIED")
        lines.append("-" * 40)
        if issues:
            for i, issue in enumerate(issues[:5], 1):
                lines.append(f"{i}. {issue['issue_name'].replace('_', ' ').title()}")
                lines.append(f"   Description: {issue['description']}")
                lines.append(f"   Frequency: {issue.get('frequency', 'N/A')} mention(s)")
                lines.append(f"   Severity: {issue['severity'].title()}")
                lines.append(f"   Example: \"{issue['example_quote']}\"")
                lines.append("")
        else:
            lines.append("No significant issues identified.")
            lines.append("")

        # Theme Analysis
        lines.append("THEME ANALYSIS")
        lines.append("-" * 40)

        if themes.get('themes'):
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

        # Recommendations
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
        lines.append(f"Estimated Cost: ${metadata['token_usage']['estimated_cost']:.4f}")
        lines.append(f"Processing Time: {metadata['analysis_time_seconds']}s")
        lines.append("")

        lines.append("=" * 80)
        lines.append("Analysis Complete")
        lines.append("=" * 80)

        return "\n".join(lines)


# Flask app setup
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["*"]}}, supports_credentials=False)

@app.route('/api/analyze-html', methods=['POST'])
def analyze_html():
    """API endpoint for analyzing HTML content."""
    try:
        # Get HTML from request
        data = request.get_json(force=True)
        html_content = data.get('html', '')
        product_type = data.get('product_type', 'product')

        if not html_content:
            return jsonify({'error': 'No HTML content provided'}), 400

        # Use the comprehensive parsing function
        reviews = parse_reviews_comprehensive(html_content)

        if not reviews:
            return jsonify({'error': 'No reviews found in HTML'}), 400

        # Limit reviews for analysis
        reviews = reviews[:ANALYSIS_REVIEW_LIMIT]

        # Initialize analyzer
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            return jsonify({'error': 'OpenAI API key not configured'}), 500

        analyzer = OpenAIReviewAnalyzer(openai_api_key)

        # Generate comprehensive analysis
        analysis_results = analyzer.generate_comprehensive_analysis(reviews, product_type)

        # Return results in format expected by React component
        response = {
            'reviews': [r['review_text'] for r in reviews],
            'sentiment': analysis_results['metrics']['sentiment_distribution'],
            'themes': analysis_results['themes'],
            'issues': analysis_results['issues'],
            'insights': analysis_results['insights'],
            'summary': analysis_results['insights'].get('executive_summary', ''),
            'analysis_metadata': {
                'total_reviews': analysis_results['metrics']['total_reviews'],
                'analysis_date': datetime.now().isoformat(),
                'product_type': product_type,
                'model_used': analysis_results['analysis_metadata']['model_used'],
                'analysis_time_seconds': analysis_results['analysis_metadata']['analysis_time_seconds'],
                'token_usage': {
                    'total_tokens': analysis_results['analysis_metadata']['token_usage']['total_tokens'],
                    'estimated_cost': round(analysis_results['analysis_metadata']['token_usage']['estimated_cost'], 4)
                }
            }
        }

        # Memory cleanup
        gc.collect()

        return jsonify(response)

    except Exception as e:
        logger.error(f"API analysis failed: {e}", exc_info=True)
        return jsonify({
            'error': 'Analysis failed',
            'message': str(e),
            'success': False
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'review-analyzer',
        'version': '3.0'
    })

def main():
    """Main execution function for standalone usage."""
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
        except Exception:
            logger.warning("Could not download NLTK 'punkt' data.")
            pass

        # Check for sample HTML file
        html_file = 'sample.html'
        if not os.path.exists(html_file):
            print(f"HTML file '{html_file}' not found.")
            return

        # Parse reviews using comprehensive function
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()

        reviews = parse_reviews_comprehensive(html_content)
        if not reviews:
            print("No reviews found or error parsing file.")
            return

        print(f"Found {len(reviews)} reviews to analyze...")

        # Get product type
        product_type = input("Enter product type (e.g., 'gaming controller', 'headphones', 'laptop'): ").strip()
        if not product_type:
            product_type = "product"

        # Initialize analyzer
        print("Initializing AI analyzer...")
        analyzer = OpenAIReviewAnalyzer(openai_api_key)

        # Generate comprehensive analysis
        print("Running AI analysis...")
        analysis_results = analyzer.generate_comprehensive_analysis(reviews, product_type)

        # Generate and display report
        print("Generating detailed report...")
        report = analyzer.create_detailed_report(analysis_results)

        print("\n" + report)

        # Export results
        with open('analysis_results.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False, default=str)

        print(f"\nAnalysis complete! Results exported to analysis_results.json")
        print(f"Total cost: ${analysis_results['analysis_metadata']['token_usage']['estimated_cost']:.4f}")

    except Exception as e:
        logger.error(f"Standalone analysis failed: {e}")
        print(f"Analysis failed: {e}")

if __name__ == "__main__":
    # Check if running as Flask app or standalone
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'standalone':
        main()
    else:
        port = int(os.getenv("PORT", 5000))
        app.run(debug=False, port=port, host="0.0.0.0")