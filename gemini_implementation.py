import json
import re
import os
import asyncio
import time
import logging
from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

import aiohttp
import nltk
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
from textstat import flesch_reading_ease
from transformers import pipeline

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Set up logging to display informational messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Main Class for Review Analysis ---
class OpenAIHybridReviewAnalyzer:
    """
    Analyzes product reviews using a hybrid approach, combining local sentiment
    analysis with OpenAI's powerful language models for deep insights.
    """

    def __init__(self, openai_api_key: str):
        """
        Initializes the analyzer, setting up the OpenAI client and loading local models.
        """
        self.device = -1
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.async_openai_client = AsyncOpenAI(api_key=openai_api_key)
        
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=self.device
            )
            logger.info(f"Local sentiment model loaded successfully on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load preferred sentiment model: {e}. Falling back to default.")
            self.sentiment_analyzer = pipeline("sentiment-analysis", device=self.device)
            
        self.token_usage = {'total_tokens': 0, 'estimated_cost': 0.0}

    async def batch_sentiment_analysis(self, reviews: List[Dict]) -> List[Dict]:
        """
        Performs sentiment analysis on a batch of reviews using the local model.
        """
        logger.info("📊 Running batch sentiment analysis...")
        sentiments = []
        for i, review in enumerate(reviews):
            if review['review_text'].strip():
                try:
                    text = review['review_text'][:512]
                    result = self.sentiment_analyzer(text)[0]
                    sentiments.append({
                        'review_index': i,
                        'sentiment': result['label'].lower(),
                        'confidence': result['score'],
                        'review_rating': review.get('review_rating', 3)
                    })
                except Exception as e:
                    logger.warning(f"Sentiment analysis failed for review {i}: {e}")
                    sentiments.append({
                        'review_index': i,
                        'sentiment': 'neutral',
                        'confidence': 0.5,
                        'review_rating': review.get('review_rating', 3)
                    })
        return sentiments

    async def extract_issues_with_llm(self, reviews: List[Dict], product_type: str) -> List[Dict]:
        """
        Uses OpenAI's language model to identify and categorize specific issues
        from negative reviews.
        """
        logger.info("🤖 Extracting issues with LLM...")
        
        negative_reviews = [
            r for r in reviews if r.get('review_rating', 3) <= 2 or
            (r.get('review_rating', 3) == 3 and 'negative' in self.sentiment_analyzer(r['review_text'][:512])[0]['label'].lower())
        ]

        if not negative_reviews:
            return []

        negative_reviews = negative_reviews[:20]
        review_texts = [
            f"Review {i+1} (Rating: {r.get('review_rating', 'N/A')}): {r['review_text'][:500]}"
            for i, r in enumerate(negative_reviews)
        ]
        combined_text = "\n\n".join(review_texts)

        # **Restored detailed prompt with a clear example format**
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
            response = await self.async_openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000,
            )
            
            self.token_usage['total_tokens'] += response.usage.total_tokens
            self.token_usage['estimated_cost'] += response.usage.total_tokens * 0.00015 / 1000

            response_content = response.choices[0].message.content.strip()
            if response_content.startswith("```json"):
                response_content = response_content[7:-3].strip()
            
            issues = json.loads(response_content)
            
            for issue in issues:
                issue['type'] = self._categorize_issue(issue.get('issue_name', ''))
                issue['review_count'] = len(negative_reviews)
                
            return issues
            
        except Exception as e:
            logger.error(f"LLM issue extraction failed: {e}")
            return [] # Return empty list on failure

    def _categorize_issue(self, issue_name: str) -> str:
        issue_name = issue_name.lower()
        categories = {
            'connectivity': ['connection', 'bluetooth', 'wifi'],
            'build_quality': ['build', 'cheap', 'material'],
            'performance': ['slow', 'lag', 'performance'],
            'battery': ['battery', 'charge', 'power'],
            'durability': ['break', 'broken', 'durability'],
            'comfort': ['comfort', 'ergonomic'],
            'functionality': ['work', 'function', 'button']
        }
        for category, keywords in categories.items():
            if any(keyword in issue_name for keyword in keywords):
                return category
        return 'other'

    async def analyze_themes_with_llm(self, reviews: List[Dict], product_type: str) -> Dict[str, Any]:
        """
        Identifies and analyzes the main themes discussed in the reviews.
        """
        logger.info("🎯 Analyzing themes with LLM...")
        
        sample_reviews = self._get_representative_sample(reviews, 15)
        sample_texts = [
            f"Review {i+1} (Rating: {r.get('review_rating', 'N/A')}): {r['review_text'][:400]}"
            for i, r in enumerate(sample_reviews)
        ]
        combined_text = "\n\n".join(sample_texts)

        # **Restored detailed two-step prompts**
        theme_prompt = f"""Analyze these {product_type} reviews and identify the main themes/aspects customers discuss.

Return 8-10 specific themes relevant to this product type. Focus on concrete features and aspects, not general sentiment.

Reviews:
{combined_text}

Return as JSON array of theme names:
["theme1", "theme2", "theme3", ...]"""

        try:
            theme_response = await self.async_openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": theme_prompt}],
                temperature=0.2,
                max_tokens=500,
            )
            response_content = theme_response.choices[0].message.content.strip()
            if response_content.startswith("```json"):
                response_content = response_content[7:-3].strip()
            themes = json.loads(response_content)

            sentiment_prompt = f"""For each theme, analyze the sentiment in these {product_type} reviews:

Themes to analyze: {themes}

Reviews:
{combined_text}

For each theme, provide:
1. Number of mentions
2. Overall sentiment (positive/negative/neutral)
3. Confidence score (0-1)
4. Key phrases/examples
5. A direct example quote from the reviews

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
            
            sentiment_response = await self.async_openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": sentiment_prompt}],
                temperature=0.1,
                max_tokens=2000,
            )
            
            total_tokens = theme_response.usage.total_tokens + sentiment_response.usage.total_tokens
            self.token_usage['total_tokens'] += total_tokens
            self.token_usage['estimated_cost'] += total_tokens * 0.00015 / 1000

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
            return {}

    def _get_representative_sample(self, reviews: List[Dict], count: int) -> List[Dict]:
        if len(reviews) <= count:
            return reviews
        
        reviews.sort(key=lambda r: len(r['review_text']), reverse=True)
        
        high_rated = [r for r in reviews if r.get('review_rating', 3) >= 4]
        low_rated = [r for r in reviews if r.get('review_rating', 3) <= 2]
        mid_rated = [r for r in reviews if r.get('review_rating', 3) == 3]
        
        sample = high_rated[:count//3] + low_rated[:count//3] + mid_rated[:count//3]
        
        remaining = count - len(sample)
        if remaining > 0:
            other_reviews = [r for r in reviews if r not in sample]
            sample.extend(other_reviews[:remaining])
            
        return sample

    async def generate_insights_with_llm(self, themes: Dict, issues: List[Dict], metrics: Dict, product_type: str) -> Dict:
        """
        Generates a high-level summary and actionable recommendations.
        """
        logger.info("💡 Generating insights with LLM...")
        
        # **Restored detailed context and prompt format**
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
5. A list of 2-3 key insights discovered from the data.

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
            response = await self.async_openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500,
            )
            
            self.token_usage['total_tokens'] += response.usage.total_tokens
            self.token_usage['estimated_cost'] += response.usage.total_tokens * 0.00015 / 1000

            response_content = response.choices[0].message.content.strip()
            if response_content.startswith("```json"):
                response_content = response_content[7:-3].strip()
        
            return json.loads(response_content)
            
        except Exception as e:
            logger.error(f"LLM insights generation failed: {e}")
            return {}

    def _calculate_enhanced_metrics(self, reviews: List[Dict], sentiments: List[Dict]) -> Dict[str, Any]:
        ratings = [r.get('review_rating', 3) for r in reviews]
        review_lengths = [len(r['review_text']) for r in reviews if r['review_text'].strip()]
        
        sentiment_counts = Counter(s['sentiment'] for s in sentiments)
        
        avg_reading_ease = 0
        if review_lengths:
            try:
                combined_text = " ".join([r['review_text'] for r in reviews[:10]])
                avg_reading_ease = flesch_reading_ease(combined_text)
            except:
                avg_reading_ease = 0
        
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

    async def generate_comprehensive_analysis(self, reviews: List[Dict], product_type: str = "product") -> Dict[str, Any]:
        """
        Orchestrates the full analysis pipeline, ensuring dependencies are met.
        """
        logger.info(f"🚀 Starting hybrid analysis for {len(reviews)} reviews...")
        start_time = time.time()

        # Step 1: Run independent data-gathering tasks concurrently.
        # These functions don't depend on each other, so they can run in parallel.
        sentiments, issues, themes = await asyncio.gather(
            self.batch_sentiment_analysis(reviews),
            self.extract_issues_with_llm(reviews, product_type),
            self.analyze_themes_with_llm(reviews, product_type)
        )

        # Step 2: Calculate metrics. This depends on the 'sentiments' result from Step 1.
        metrics = self._calculate_enhanced_metrics(reviews, sentiments)

        # Step 3: Generate the final insights. This depends on themes, issues, and metrics from the previous steps.
        final_insights = await self.generate_insights_with_llm(themes, issues, metrics, product_type)

        analysis_time = time.time() - start_time

        return {
            'themes': themes,
            'issues': issues,
            'metrics': metrics,
            'insights': final_insights,
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
        # **Restored comprehensive report structure**
        themes = analysis_results.get('themes', {})
        issues = analysis_results.get('issues', [])
        metrics = analysis_results.get('metrics', {})
        insights = analysis_results.get('insights', {})
        metadata = analysis_results.get('analysis_metadata', {})
        
        lines = []
        lines.append("=" * 80)
        lines.append("📊 AI-ENHANCED PRODUCT REVIEW ANALYSIS")
        lines.append("=" * 80)
        lines.append(f"Product: {metadata.get('product_type', 'N/A').title()}")
        lines.append(f"Reviews Analyzed: {metadata.get('total_reviews', 0)}")
        lines.append(f"Analysis Date: {metadata.get('analysis_date', 'N/A')}")
        lines.append(f"Model: {metadata.get('model_used', 'N/A')}")
        lines.append(f"Processing Time: {metadata.get('analysis_time_seconds', 0)}s")
        lines.append(f"API Cost: ${metadata.get('token_usage', {}).get('estimated_cost', 0):.4f}")
        lines.append("")
        
        lines.append("📋 EXECUTIVE SUMMARY")
        lines.append("-" * 40)
        lines.append(insights.get('executive_summary', 'No summary available.'))
        lines.append("")
        
        if insights.get('key_insights'):
            lines.append("💡 KEY INSIGHTS")
            lines.append("-" * 40)
            for insight in insights['key_insights']:
                lines.append(f"• {insight}")
            lines.append("")
        
        lines.append("📈 KEY METRICS")
        lines.append("-" * 40)
        lines.append(f"Average Rating: {metrics.get('average_rating', 0):.1f}/5")
        lines.append(f"Total Reviews: {metrics.get('total_reviews', 0)}")
        lines.append(f"Average Review Length: {metrics.get('average_review_length', 0):.0f} characters")
        lines.append(f"Analysis Quality: {metrics.get('analysis_quality', 'N/A').title()}")
        lines.append("")
        
        if metrics.get('sentiment_distribution'):
            lines.append("Sentiment Distribution:")
            total_sentiment = sum(metrics['sentiment_distribution'].values())
            emoji_map = {'positive': '😊', 'neutral': '😐', 'negative': '😞'}
            for sentiment, count in metrics['sentiment_distribution'].items():
                percentage = (count / total_sentiment * 100) if total_sentiment > 0 else 0
                lines.append(f"  {emoji_map.get(sentiment, '🤔')} {sentiment.title()}: {count} ({percentage:.1f}%)")
            lines.append("")

        lines.append("🔴 TOP ISSUES IDENTIFIED (AI-POWERED)")
        lines.append("-" * 40)
        if issues:
            for i, issue in enumerate(issues[:5], 1):
                lines.append(f"{i}. {issue.get('issue_name', 'N/A').title()}")
                lines.append(f"   Description: {issue.get('description', 'N/A')}")
                lines.append(f"   Frequency: {issue.get('frequency', 'N/A')} mention(s)")
                lines.append(f"   Severity: {issue.get('severity', 'N/A').title()}")
                lines.append(f"   Example: \"{issue.get('example_quote', 'N/A')}\"")
                lines.append("")
        else:
            lines.append("No significant issues identified.")
            lines.append("")
        
        if themes.get('themes'):
            lines.append("🎯 THEME ANALYSIS (AI-POWERED)")
            lines.append("-" * 40)
            for theme_name, data in themes['themes'].items():
                lines.append(f"📌 {theme_name.replace('_', ' ').title()}")
                lines.append(f"   Mentions: {data.get('mentions', 'N/A')}")
                lines.append(f"   Sentiment: {data.get('sentiment', 'N/A').title()}")
                lines.append(f"   Key Phrases: {', '.join(data.get('key_phrases', []))}")
                lines.append(f"   Example: \"{data.get('example_quote', 'N/A')}\"")
                lines.append("")
        
        if insights.get('recommendations'):
            lines.append("💡 AI-POWERED RECOMMENDATIONS")
            lines.append("-" * 40)
            for i, rec in enumerate(insights['recommendations'], 1):
                lines.append(f"{i}. {rec.get('recommendation', 'N/A')}")
                lines.append(f"   Priority: {rec.get('priority', 'N/A').title()}")
                lines.append(f"   Impact: {rec.get('impact', 'N/A')}")
                lines.append(f"   Rationale: {rec.get('rationale', 'N/A')}")
                lines.append("")
        
        lines.append("=" * 80)
        lines.append("✅ Analysis Complete")
        lines.append("=" * 80)
        
        return "\n".join(lines)

def parse_reviews_from_file(file_path: str) -> Optional[List[Dict]]:
    """
    Parses product reviews from an HTML file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
        
    soup = BeautifulSoup(html_content, 'html.parser')
    reviews = []
    
    # Define selectors to find review elements
    review_selectors = ['li.review-item', '.review', '[data-testid="review"]']
    for selector in review_selectors:
        review_elements = soup.select(selector)
        if review_elements:
            break
    
    if not review_elements:
        logger.warning("No review elements found in the HTML file.")
        return []

    for element in review_elements:
        review_data = {'review_text': '', 'review_rating': None}
        
        # Extract review text
        text_element = element.select_one('p.pre-white-space, .review-text, .content')
        if text_element:
            review_data['review_text'] = text_element.get_text(strip=True)
            
        # Extract review rating
        rating_element = element.select_one('p.visually-hidden')
        if rating_element:
            match = re.search(r'Rated (\d(\.\d)?)', rating_element.text)
            if match:
                review_data['review_rating'] = float(match.group(1))
                
        if review_data['review_text']:
            reviews.append(review_data)
            
    logger.info(f"Successfully parsed {len(reviews)} reviews from {file_path}")
    return reviews

async def main():
    """
    The main function to run the review analysis process.
    """
    # Get OpenAI API key from environment variables or user input
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        openai_api_key = input("Enter your OpenAI API key: ").strip()
    if not openai_api_key:
        logger.error("OpenAI API key is required to run this script.")
        return

    # Download necessary NLTK data
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except Exception as e:
        logger.warning(f"Could not download NLTK data: {e}")

    # Parse reviews from the sample HTML file
    reviews = parse_reviews_from_file('sample.html')
    if not reviews:
        logger.error("No reviews found or failed to parse the file.")
        return

    product_type = input("Enter product type (e.g., 'gaming controller'): ").strip() or "gaming controller"
    
    # Initialize and run the analysis
    analyzer = OpenAIHybridReviewAnalyzer(openai_api_key)
    analysis_results = await analyzer.generate_comprehensive_analysis(reviews, product_type)
    
    # Generate and display the final report
    report = analyzer.create_detailed_report(analysis_results)
    print("\n" + report)
    
    # Save the detailed analysis results to a JSON file
    output_filename = 'openai_enhanced_analysis.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=4, ensure_ascii=False)
        
    logger.info(f"Analysis complete. Results saved to {output_filename}")
    logger.info(f"Total cost: ${analysis_results['analysis_metadata']['token_usage']['estimated_cost']:.4f}")

if __name__ == "__main__":
    asyncio.run(main())