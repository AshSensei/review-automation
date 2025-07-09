import json
import re
import os
from collections import Counter, defaultdict
from bs4 import BeautifulSoup
from transformers import pipeline
import logging
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from textstat import flesch_reading_ease
from openai import OpenAI
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import time
from dotenv import load_dotenv
import os

load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIHybridReviewAnalyzer:
    def __init__(self, openai_api_key: str):
        """Initialize the hybrid analyzer with OpenAI integration."""
        self.device = -1 #testing device only has cpu
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize local models for bulk processing
        try:
            # Keep local sentiment for bulk analysis
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=self.device
            )
            
            logger.info(f"Local models loaded successfully on device: {self.device}")
        except Exception as e:
            logger.error(f"Error loading local models: {e}")
            # Fallback to basic model
            self.sentiment_analyzer = pipeline("sentiment-analysis", device=self.device)
        
        # Token usage tracking
        self.token_usage = {
            'total_tokens': 0,
            'estimated_cost': 0.0
        }
    
    def batch_sentiment_analysis(self, reviews: List[Dict]) -> List[Dict]:
        """Perform bulk sentiment analysis using local models."""
        print("📊 Running batch sentiment analysis...")
        
        sentiments = []
        for i, review in enumerate(reviews):
            if review['review_text'].strip():
                try:
                    # Truncate for local model
                    text = review['review_text'][:512]
                    sentiment = self.sentiment_analyzer(text)[0]
                    sentiments.append({
                        'review_index': i,
                        'sentiment': sentiment['label'],
                        'confidence': sentiment['score'],
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
    
    def extract_issues_with_llm(self, reviews: List[Dict], product_type: str) -> List[Dict]:
        """Extract specific issues using OpenAI for negative reviews."""
        print("🤖 Extracting issues with LLM...")
        
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
        
        prompt = f"""Analyze these negative {product_type} reviews and extract specific product issues.

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
            
            # Track token usage
            self.token_usage['total_tokens'] += response.usage.total_tokens
            self.token_usage['estimated_cost'] += response.usage.total_tokens * 0.00015 / 1000  # GPT-4o-mini pricing

            # Get the response content and strip markdown
            response_content = response.choices[0].message.content.strip()
            if response_content.startswith("```json"):
                response_content = response_content[7:-3].strip()

            issues = json.loads(response_content)
            
            # Add review context
            for issue in issues:
                issue['type'] = self.categorize_issue(issue['issue_name'])
                issue['review_count'] = len(negative_reviews)
            
            return issues
            
        except Exception as e:
            logger.error(f"LLM issue extraction failed: {e}")
            return self.fallback_issue_extraction(negative_reviews)
    
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
        """Analyze themes using OpenAI with smart sampling."""
        print("🎯 Analyzing themes with LLM...")
        
        # Smart sampling: get diverse representative reviews
        sample_reviews = self.get_representative_sample(reviews, 15)
        
        # Prepare sample text
        sample_texts = []
        for i, review in enumerate(sample_reviews):
            rating = review.get('review_rating', 'N/A')
            text = review['review_text'][:400]  # Truncate for token efficiency
            sample_texts.append(f"Review {i+1} (Rating: {rating}): {text}")
        
        combined_text = "\n\n".join(sample_texts)
        
        # Two-step process: 1) Discover themes, 2) Analyze sentiment
        
        # Step 1: Theme Discovery
        theme_prompt = f"""Analyze these {product_type} reviews and identify the main themes/aspects customers discuss.

Return 8-10 specific themes relevant to this product type. Focus on concrete features and aspects, not general sentiment.

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
            sentiment_prompt = f"""For each theme, analyze the sentiment in these {product_type} reviews:

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
            self.token_usage['estimated_cost'] += total_tokens * 0.00015 / 1000

            # Get the response content and strip markdown
            response_content = sentiment_response.choices[0].message.content.strip()
            if response_content.startswith("```json"):
                response_content = response_content[7:-3].strip()
                
            theme_analysis = json.loads(response_content)
            
            return {
                'themes': theme_analysis,
                'discovered_themes': themes,
                'sample_size': len(sample_reviews)
            }
            
        except Exception as e:
            logger.error(f"LLM theme analysis failed: {e}")
            return self.fallback_theme_analysis(reviews, product_type)
    
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
        """Calculate comprehensive metrics combining local and LLM analysis."""
        
        ratings = [r.get('review_rating', 3) for r in reviews]
        review_lengths = [len(r['review_text']) for r in reviews if r['review_text'].strip()]
        
        # Sentiment distribution from local analysis
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
        """Generate comprehensive analysis using hybrid approach."""
        
        print(f"🚀 Starting hybrid analysis for {len(reviews)} reviews...")
        start_time = time.time()
        
        # Step 1: Local sentiment analysis (free/fast)
        sentiments = self.batch_sentiment_analysis(reviews)
        
        # Step 2: LLM-powered issue extraction (focused on negatives)
        issues = self.extract_issues_with_llm(reviews, product_type)
        
        # Step 3: LLM-powered theme analysis (smart sampling)
        themes = self.analyze_themes_with_llm(reviews, product_type)
        
        # Step 4: Enhanced metrics calculation
        metrics = self.calculate_enhanced_metrics(reviews, sentiments)
        
        # Step 5: LLM-powered insights generation
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
                'model_used': 'gpt-4o-mini + local models',
                'analysis_time_seconds': round(analysis_time, 2),
                'token_usage': self.token_usage.copy()
            }
        }
    
    def create_detailed_report(self, analysis_results: Dict[str, Any]) -> str:
        """Create a detailed, actionable report."""
        
        themes = analysis_results['themes']
        issues = analysis_results['issues']
        metrics = analysis_results['metrics']
        insights = analysis_results['insights']
        metadata = analysis_results['analysis_metadata']
        
        lines = []
        lines.append("=" * 80)
        lines.append("📊 OPENAI-ENHANCED PRODUCT REVIEW ANALYSIS")
        lines.append("=" * 80)
        lines.append(f"Product: {metadata['product_type'].title()}")
        lines.append(f"Reviews Analyzed: {metadata['total_reviews']}")
        lines.append(f"Analysis Date: {metadata['analysis_date']}")
        lines.append(f"Model: {metadata['model_used']}")
        lines.append(f"Processing Time: {metadata['analysis_time_seconds']}s")
        lines.append(f"API Cost: ${metadata['token_usage']['estimated_cost']:.4f}")
        lines.append("")
        
        # Executive Summary
        lines.append("📋 EXECUTIVE SUMMARY")
        lines.append("-" * 40)
        lines.append(insights.get('executive_summary', 'Analysis completed'))
        lines.append("")
        
        # Key Insights
        if insights.get('key_insights'):
            lines.append("💡 KEY INSIGHTS")
            lines.append("-" * 40)
            for insight in insights['key_insights']:
                lines.append(f"• {insight}")
            lines.append("")
        
        # Key Metrics
        lines.append("📈 KEY METRICS")
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
            emoji = "😊" if "pos" in sentiment.lower() else "😞" if "neg" in sentiment.lower() else "😐"
            lines.append(f"  {emoji} {sentiment}: {count} ({percentage:.1f}%)")
        lines.append("")
        
        # Top Issues (LLM-powered)
        lines.append("🔴 TOP ISSUES IDENTIFIED (AI-POWERED)")
        lines.append("-" * 40)
        if issues:
            for i, issue in enumerate(issues[:5], 1):
                lines.append(f"{i}. {issue['issue_name'].replace('_', ' ').title()}")
                lines.append(f"   Description: {issue['description']}")
                lines.append(f"   Frequency: {issue['frequency']} mentions")
                lines.append(f"   Severity: {issue['severity'].title()}")
                lines.append(f"   Example: \"{issue['example_quote']}\"")
                lines.append("")
        else:
            lines.append("No significant issues identified")
            lines.append("")
        
        # Theme Analysis (LLM-powered)
        lines.append("🎯 THEME ANALYSIS (AI-POWERED)")
        lines.append("-" * 40)
        
        if themes.get('themes'):
            for theme_name, theme_data in themes['themes'].items():
                lines.append(f"📌 {theme_name.replace('_', ' ').title()}")
                lines.append(f"   Mentions: {theme_data['mentions']}")
                lines.append(f"   Sentiment: {theme_data['sentiment'].title()}")
                lines.append(f"   Confidence: {theme_data['confidence']:.2f}")
                
                if theme_data.get('key_phrases'):
                    lines.append(f"   Key Phrases: {', '.join(theme_data['key_phrases'])}")
                
                if theme_data.get('example_quote'):
                    lines.append(f"   Example: \"{theme_data['example_quote']}\"")
                lines.append("")
        
        # Recommendations (LLM-powered)
        lines.append("💡 AI-POWERED RECOMMENDATIONS")
        lines.append("-" * 40)
        
        if insights.get('recommendations'):
            for i, rec in enumerate(insights['recommendations'], 1):
                priority_emoji = "🔥" if rec['priority'] == 'high' else "⚡" if rec['priority'] == 'medium' else "💡"
                lines.append(f"{i}. {priority_emoji} {rec['recommendation']}")
                lines.append(f"   Priority: {rec['priority'].title()}")
                lines.append(f"   Impact: {rec['impact']}")
                lines.append(f"   Rationale: {rec['rationale']}")
                lines.append("")
        
        # Analysis Statistics
        lines.append("📊 ANALYSIS STATISTICS")
        lines.append("-" * 40)
        lines.append(f"Tokens Used: {metadata['token_usage']['total_tokens']:,}")
        lines.append(f"Estimated Cost: ${metadata['token_usage']['estimated_cost']:.4f}")
        lines.append(f"Processing Time: {metadata['analysis_time_seconds']}s")
        lines.append("")
        
        lines.append("=" * 80)
        lines.append("✅ AI-Enhanced Analysis Complete")
        lines.append("=" * 80)
        
        return "\n".join(lines)

# HTML parsing function (keep from original)
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

def main():
    """Main execution function."""
    try:
        # Get OpenAI API key
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            openai_api_key = input("Enter your OpenAI API key: ").strip()
        
        if not openai_api_key:
            print("❌ OpenAI API key is required")
            return
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except:
            pass
        
        # Parse reviews
        reviews = parse_reviews_from_file('sample.html')
        if not reviews:
            print("❌ No reviews found or error parsing file.")
            return
        
        print(f"📚 Found {len(reviews)} reviews to analyze...")
        
        # Get product type
        product_type = input("Enter product type (e.g., 'gaming controller', 'headphones', 'laptop'): ").strip()
        if not product_type:
            product_type = "gaming controller"
        
        # Initialize hybrid analyzer
        print("🤖 Initializing OpenAI-enhanced analyzer...")
        analyzer = OpenAIHybridReviewAnalyzer(openai_api_key)
        
        # Generate comprehensive analysis
        print("🔍 Running AI-enhanced analysis...")
        analysis_results = analyzer.generate_comprehensive_analysis(reviews, product_type)
        
        # Generate and display report
        print("📊 Generating detailed report...")
        report = analyzer.create_detailed_report(analysis_results)
        
        print("\n" + report)
        
        # Export results
        with open('openai_enhanced_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n✅ Analysis complete! Results exported to openai_enhanced_analysis.json")
        print(f"💰 Total cost: ${analysis_results['analysis_metadata']['token_usage']['estimated_cost']:.4f}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"❌ Analysis failed: {e}")

if __name__ == "__main__":
    main()