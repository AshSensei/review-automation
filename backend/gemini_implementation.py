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
    Analyzes product reviews using a multi-layered hybrid approach, combining local
    sentiment analysis with advanced, multi-step AI insights from OpenAI.
    """

    def __init__(self, openai_api_key: str):
        self.device = -1  # Use -1 for CPU, or 0 for the first GPU
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

    async def _json_loader(self, response_content: str) -> Any:
        if response_content.startswith("```json"):
            response_content = response_content[7:-3].strip()
        elif response_content.startswith("```"):
            response_content = response_content[3:-3].strip()
        return json.loads(response_content)

    async def _run_llm_request(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.1) -> Any:
        try:
            response = await self.async_openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            self.token_usage['total_tokens'] += response.usage.total_tokens
            self.token_usage['estimated_cost'] += response.usage.total_tokens * 0.00015 / 1000
            return await self._json_loader(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"An LLM request failed: {e}")
            return None

    async def batch_sentiment_analysis(self, reviews: List[Dict]) -> List[Dict]:
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
        logger.info("🤖 Extracting issues from negative reviews...")
        negative_reviews = [
            r for r in reviews if r.get('review_rating', 3) <= 2
        ]
        if not negative_reviews:
            return []

        negative_reviews = negative_reviews[:20]
        review_texts = [
            f"Review {i+1} (Rating: {r.get('review_rating', 'N/A')}): {r['review_text'][:500]}"
            for i, r in enumerate(negative_reviews)
        ]
        prompt = f"""Analyze these negative {product_type} reviews and extract specific product issues.

For each issue found, provide:
1. Issue name (brief, specific)
2. Description (what exactly is wrong)
3. Frequency (how many reviews mention this)
4. Severity (high/medium/low based on impact)
5. Example quote from reviews

Return as JSON array:
{{
  "issues": [
    {{
      "issue_name": "connection_drops",
      "description": "Device frequently disconnects during use",
      "frequency": 3,
      "severity": "high",
      "example_quote": "keeps disconnecting every few minutes"
    }}
  ]
}} 

Reviews:
{chr(10).join(review_texts)}"""
        issues_data = await self._run_llm_request(prompt)
        if not issues_data:
            return []
        return issues_data

    async def analyze_themes_with_llm(self, reviews: List[Dict], product_type: str) -> Dict[str, Any]:
        logger.info("🎯 Analyzing themes with enhanced sentiment analysis...")
        sample_reviews = self._get_representative_sample(reviews, 15)
        sample_texts = [
            f"Review {i+1} (Rating: {r.get('review_rating', 'N/A')}): {r['review_text'][:400]}"
            for i, r in enumerate(sample_reviews)
        ]
        combined_text = "\n\n".join(sample_texts)

        theme_prompt = f"""Analyze these {product_type} reviews and identify the main themes/aspects customers discuss.

Return 8-10 specific themes relevant to this product type. Focus on concrete features and aspects, not general sentiment.

Reviews:
{combined_text}

Return as JSON array:
["theme1", "theme2", "theme3", ...]"""

        themes = await self._run_llm_request(theme_prompt, max_tokens=500, temperature=0.2)
        if not themes:
            return {}

        sentiment_prompt = f"""For each theme, analyze the sentiment in these {product_type} reviews:

Themes to analyze: {themes}

Reviews:
{combined_text}

For each theme, provide:
1. Number of mentions
2. Overall sentiment (positive/negative/neutral)
3. Confidence score (0-1)
4. Key phrases/examples
5. A direct example quote

Return as JSON:
{{
  "theme_name": {{
    "mentions": 5,
    "sentiment": "positive",
    "confidence": 0.8,
    "key_phrases": ["works great", "very responsive"],
    "example_quote": "buttons are very responsive"
  }}
}}"""
        analysis = await self._run_llm_request(sentiment_prompt)
        return {'themes': analysis, 'discovered_themes': themes, 'sample_size': len(sample_reviews)}

    async def generate_insights_with_llm(self, themes: Dict, issues: List[Dict], metrics: Dict, product_type: str) -> Dict:
        logger.info("💡 Generating insights and recommendations...")
        context = f"""Product Type: {product_type}
Total Reviews: {metrics['total_reviews']}
Average Rating: {metrics['average_rating']:.1f}
Sentiment Distribution: {metrics['sentiment_distribution']}

Top Themes:
{json.dumps(themes.get('themes', {}), indent=2)}

Key Issues:
{json.dumps(issues[:5] if isinstance(issues, list) else [], indent=2)}"""

        prompt = f"""Based on this product analysis, provide:
1. Executive summary
2. Top 5 actionable recommendations
3. Priority (high/medium/low)
4. Business impact
5. 2-3 key insights

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
  "key_insights": ["insight1", "insight2"]
}}"""
        return await self._run_llm_request(prompt, max_tokens=1500, temperature=0.3)

    async def perform_root_cause_analysis(self, issues: List[Dict], reviews: List[Dict]) -> Optional[Dict]:
        """Performs a deeper dive to find the root cause of the top negative issue."""
        if not issues or not isinstance(issues, list):
            return None
        
        # Filter only dictionaries with expected keys
        filtered_issues = [i for i in issues if isinstance(i, dict) and 'severity' in i and 'issue_name' in i]
        if not filtered_issues:
            return None
        
        top_issue = sorted(
            filtered_issues,
            key=lambda x: x.get('severity', 'low') == 'high',
            reverse=True
        )[0]
        
        issue_name = top_issue.get('issue_name', 'Unknown Issue')
        logger.info(f"🔬 Performing root cause analysis for top issue: '{issue_name}'...")

        review_texts = "\n".join([r['review_text'] for r in reviews])
        prompt = f"""Based on the provided reviews, perform a root cause analysis for the issue of '{issue_name}'. 
    What specific aspects are users complaining about? Categorize their complaints into 2-3 likely root causes.
    For each cause, provide a brief description and a supporting quote.

    Reviews:
    {review_texts[:4000]} 

    Return as a JSON object:
    {{
    "analyzed_issue": "{issue_name}",
    "possible_causes": [
        {{"cause": "Component Quality", "description": "The physical parts seem to be low-grade.", "supporting_quote": "..."}}
    ]
    }}"""
        return await self._run_llm_request(prompt)


    async def identify_user_personas(self, reviews: List[Dict]) -> Optional[List]:
        logger.info("👥 Identifying user personas...")
        sample_texts = "\n\n".join([f"Review: {r['review_text'][:400]}" for r in self._get_representative_sample(reviews, 20)])
        prompt = f"""From these reviews, identify 2-3 user personas. For each, give:
- Name (e.g., 'The Power User')
- Priorities
- Product aspects they care about

Return as JSON:
[
  {{
    "name": "The Competitive Gamer",
    "priorities": ["Response Time", "Stick Accuracy"],
    "relevant_themes": ["Positive: Response Time", "Negative: Stick Wear"]
  }}
]"""
        return await self._run_llm_request(prompt)

    async def generate_comprehensive_analysis(self, reviews: List[Dict], product_type: str = "product") -> Dict[str, Any]:
        logger.info(f"🚀 Starting comprehensive analysis for {len(reviews)} reviews...")
        start_time = time.time()

        sentiments, issues, themes = await asyncio.gather(
            self.batch_sentiment_analysis(reviews),
            self.extract_issues_with_llm(reviews, product_type),
            self.analyze_themes_with_llm(reviews, product_type)
        )

        metrics = self._calculate_enhanced_metrics(reviews, sentiments)
        final_insights = await self.generate_insights_with_llm(themes, issues, metrics, product_type)
        root_cause_analysis, user_personas = await asyncio.gather(
            self.perform_root_cause_analysis(issues, reviews),
            self.identify_user_personas(reviews)
        )

        return {
            'themes': themes,
            'issues': issues,
            'metrics': metrics,
            'insights': final_insights,
            'root_cause_analysis': root_cause_analysis,
            'user_personas': user_personas,
            'analysis_metadata': {
                'total_reviews': len(reviews),
                'analysis_date': datetime.now().isoformat(),
                'product_type': product_type,
                'model_used': 'gpt-4o-mini + local models',
                'analysis_time_seconds': round(time.time() - start_time, 2),
                'token_usage': self.token_usage.copy()
            }
        }

    def _get_representative_sample(self, reviews: List[Dict], count: int) -> List[Dict]:
        if len(reviews) <= count:
            return reviews
        reviews.sort(key=lambda r: len(r['review_text']), reverse=True)
        high_rated = [r for r in reviews if r.get('review_rating', 3) >= 4]
        low_rated = [r for r in reviews if r.get('review_rating', 3) <= 2]
        mid_rated = [r for r in reviews if r.get('review_rating', 3) == 3]
        sample = high_rated[:count // 3] + low_rated[:count // 3] + mid_rated[:count // 3]
        remaining = count - len(sample)
        if remaining > 0:
            others = [r for r in reviews if r not in sample]
            sample.extend(others[:remaining])
        return sample

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
    """The main function to run the complete analysis process."""
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        openai_api_key = input("Enter your OpenAI API key: ").strip()
    if not openai_api_key:
        logger.error("OpenAI API key is required.")
        return

    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except Exception as e:
        logger.warning(f"Could not download NLTK data: {e}")

    reviews = parse_reviews_from_file('sample.html')
    if not reviews:
        # The function already logs the error, so we can just exit.
        return

    product_type = input("Enter product type (e.g., 'gaming controller'): ").strip() or "gaming controller"
    
    # Initialize the new, more powerful analyzer class
    analyzer = OpenAIHybridReviewAnalyzer(openai_api_key)
    
    # This call now orchestrates all analysis steps, including the new ones
    analysis_results = await analyzer.generate_comprehensive_analysis(reviews, product_type)
    
    # This report function is updated to display the new sections
    report = analyzer.create_detailed_report(analysis_results)
    print("\n" + report)
    
    output_filename = 'advanced_product_analysis.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=4, ensure_ascii=False)
        
    logger.info(f"Analysis complete. Full results saved to {output_filename}")
    logger.info(f"Estimated cost: ${analysis_results.get('analysis_metadata', {}).get('token_usage', {}).get('estimated_cost', 0):.4f}")

if __name__ == "__main__":
    asyncio.run(main())