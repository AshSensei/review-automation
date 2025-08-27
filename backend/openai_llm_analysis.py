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
MAX_HTML_SIZE = 3 * 1024 * 1024 * 5 # 3MB
PARSE_HTML_LIMIT = 200  # max reviews to parse
ANALYSIS_REVIEW_LIMIT = 200  # max reviews to analyze
NEGATIVE_SAMPLE = 200  # negative reviews sample size
MAX_ISSUES_TO_ENRICH = 5
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_reviews_comprehensive(html_content: str) -> List[Dict[str, Any]]:
    """
    Parses HTML content from multiple sources by collecting reviews
    from a list of known good selectors and deduplicating.
    """
    try:
        if len(html_content) > MAX_HTML_SIZE:
            html_content = html_content[:MAX_HTML_SIZE]

        soup = BeautifulSoup(html_content, "html.parser")
        reviews = []

        container_selectors = [
            '[data-hook="review"]',
            '.review-item',
            '[data-testid="review"]',
            '.review',
            '.customer-review',
        ]

        all_review_elements = []
        for selector in container_selectors:
            elements = soup.select(selector)
            if elements:
                all_review_elements.extend(elements)

        if all_review_elements:
            unique_elements = list(dict.fromkeys(all_review_elements))
            review_elements = unique_elements[:PARSE_HTML_LIMIT]
        else:
            return []

        for i, element in enumerate(review_elements):
            try:
                review_data = {
                    "review_text": "",
                    "review_rating": None,
                    "review_id": f"review_{i}",
                }

                text_element = element.select_one('[data-hook="review-body"] span')
                if not text_element:
                    text_element = element.select_one('.review-text, .review-content p, .ugc-review-body')

                review_text = text_element.get_text(strip=True) if text_element else ""

                if not review_text:
                    review_text = element.get_text(strip=True)

                review_data["review_text"] = review_text[:1000]

                rating = None
                rating_element = element.select_one('i[data-hook="review-star-rating"] span.a-icon-alt')
                if rating_element:
                    rating_text = rating_element.get_text(strip=True)
                    rating_match = re.search(r"(\d+(\.\d)?)", rating_text)
                    if rating_match:
                        rating = float(rating_match.group(1))

                if rating is None:
                    generic_rating_selectors = [
                        '[aria-label*="star"]',
                        '.review-rating',
                        '.rating',
                    ]
                    for selector in generic_rating_selectors:
                        generic_rating_element = element.select_one(selector)
                        if generic_rating_element:
                            rating_text = generic_rating_element.get_text(strip=True)
                            rating_match = re.search(r"(\d+(\.\d)?)", rating_text)
                            if rating_match:
                                rating = float(rating_match.group(1))
                                break
                            
                            aria_label = generic_rating_element.get("aria-label", "")
                            aria_match = re.search(r"(\d+(\.\d)?)", aria_label)
                            if aria_match:
                                rating = float(aria_match.group(1))
                                break
                
                review_data["review_rating"] = rating
                reviews.append(review_data)

            except Exception as e:
                # logger.debug(f"Failed to parse individual review {i}: {e}")
                continue
        
        return reviews

    except Exception as e:
        # logger.error(f"Failed to parse HTML: {e}")
        return []

class OpenAIReviewAnalyzer:
    def __init__(self, openai_api_key: str):
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.token_usage = {"total_tokens": 0, "estimated_cost": 0.0}

    def _track_usage(self, usage, model_name: str):
        if not usage:
            return

        # Prices per 1M tokens for input/output
        pricing = {
            "gpt-5": (1.25, 10.00),
            "gpt-5-mini": (0.25, 2.00),
            "gpt-5-nano": (0.05, 0.40),
        }

        # Simplified cost estimation
        input_price, _ = pricing.get(model_name, (0, 0))
        tokens = usage.total_tokens
        self.token_usage["total_tokens"] += tokens
        self.token_usage["estimated_cost"] += (tokens * input_price) / 1000000

    def _parse_json_from_response(self, response: Any) -> Any:
        try:
            final_message = next(
                (item for item in response.output if item.type == "message"), None
            )
            if final_message:
                raw_output = final_message.content[0].text
                clean_json_string = re.sub(r",\s*([\}\]])", r"\1", raw_output.strip())
                content = re.sub(r"^```json|```$", "", clean_json_string)
                return json.loads(content)
            else:
                logger.error("No 'message' object found in the API response output.")
                return None
        except (IndexError, AttributeError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse JSON from response: {e}")
            return None

    def batch_sentiment_analysis(self, reviews: List[Dict]) -> List[Dict]:
        """
        Performs sentiment analysis based solely on star ratings.
        No API calls needed - this is now a simple rating-to-sentiment mapping.
        """
        logger.info("📊 Running rating-based sentiment analysis...")
        sentiments = []
        
        for i, review in enumerate(reviews):
            rating = review.get("review_rating")
            
            # Map star rating to sentiment
            if rating is None:
                # If no rating available, default to neutral
                sentiment = "neutral"
                confidence = 0.3
            elif rating >= 4:
                sentiment = "positive"
                confidence = 0.9
            elif rating <= 2:
                sentiment = "negative" 
                confidence = 0.9
            else:  # rating == 3
                sentiment = "neutral"
                confidence = 0.8
                
            sentiments.append(
                {
                    "review_index": i,
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "review_rating": rating,
                }
            )
        
        logger.info(f"Completed sentiment analysis for {len(sentiments)} reviews")
        return sentiments

    def extract_issues_with_llm(
        self, reviews: List[Dict], product_type: str, metrics: Dict, return_full_response: bool = False
    ) -> Any:
        logger.info("🤖 Extracting issues with GPT-5...")
        negative_reviews = [r for r in reviews if r.get("review_rating", 3) <= 2]
        if not negative_reviews:
            negative_reviews = reviews

        # Include metrics in the prompt for better context
        metrics_context = f"""
Product Metrics:
- Total Reviews: {metrics.get('total_reviews', 0)}
- Average Rating: {metrics.get('average_rating', 0):.1f}/5
- Sentiment Distribution: {metrics.get('sentiment_distribution', {})}
"""

        # A more defensive prompt to ensure valid JSON
        prompt = f"""
<instructions>
You are an API that only returns valid JSON. Your task is to analyze product reviews and extract key issues.

{metrics_context}

Follow this two-step process:
1.  **Analysis Step:** First, think through all the reviews provided. Identify up to 5 distinct user issues. For each issue, internally note its name, a one-sentence description, its severity ('high', 'medium', or 'low'), the number of reviews that mention it ('frequency'), and find the best single 'example_quote'.
2.  **Formatting Step:** Second, take the issues you identified in your analysis and format them into a single, valid JSON array of objects.

Your final output MUST be only the JSON array. Do not include your reasoning from Step 1.
</instructions>

Example of the required JSON output format:
[
  {{
    "issue_name": "Thumbstick Wear",
    "description": "Users report the plastic thumbsticks grind against the anti-friction rings, causing physical damage.",
    "severity": "high",
    "frequency": 3,
    "example_quote": "the thumb sticks grind against the terrible friction ring..."
  }}
]

Reviews:
""" + "\n".join(
            f"Review (Rating: {r.get('review_rating', 'N/A')}): {r['review_text']}" for r in negative_reviews
        )

        try:
            response = self.openai_client.responses.create(
                model="gpt-5", input=prompt, reasoning={"effort": "low"}
            )
            self._track_usage(response.usage, "gpt-5")

            if return_full_response:
                return response

            # The robust helper function will parse the cleaned output
            issues_data = self._parse_json_from_response(response)

            if not issues_data:
                raise ValueError("Parsing issues returned None.")

            if isinstance(issues_data, dict):
                key_with_list = next(
                    (k for k, v in issues_data.items() if isinstance(v, list)), None
                )
                issues = issues_data[key_with_list] if key_with_list else []
            else:
                issues = issues_data

            enriched_issues = []
            for issue in issues[:MAX_ISSUES_TO_ENRICH]:
                name = issue.get("issue_name", "Unnamed Issue")
                enriched_issues.append(
                    {
                        "issue_name": name,
                        "description": issue.get("description", ""),
                        "severity": issue.get("severity", "medium"),
                        "review_count": issue.get("frequency", 1),
                        "example_quote": issue.get("example_quote", ""),
                        "related_quotes": [issue.get("example_quote", "")],
                        "type": self.categorize_issue(name),
                    }
                )
            return enriched_issues
        except Exception as e:
            logger.error(f"LLM issue extraction failed: {e}")
            return [] if not return_full_response else None

    def categorize_issue(self, issue_name: str) -> str:
        """Enhanced issue categorization combining both approaches."""
        issue_name_lower = issue_name.lower()

        # Your existing categories
        categories = {
            "connectivity": ["connection", "disconnect", "bluetooth", "wifi", "pair"],
            "build_quality": ["build", "quality", "cheap", "flimsy", "material"],
            "performance": ["slow", "lag", "performance", "speed", "responsive"],
            "battery": ["battery", "charge", "power", "drain"],
            "durability": ["break", "broke", "broken", "durable", "last"],
            "comfort": ["comfort", "hurt", "pain", "ergonomic"],
            "functionality": ["work", "function", "feature", "button"],
        }

        # Additional categories from original code
        additional_categories = {
            "logistics": ["shipping", "delivery", "arrival", "package"],
            "pricing": ["price", "cost", "expensive", "value"],
            "customer_service": ["customer", "service", "support", "help"],
            "sizing": ["size", "fit", "dimension", "scale"],
        }

        # Merge categories
        all_categories = {**categories, **additional_categories}

        for category, keywords in all_categories.items():
            if any(keyword in issue_name_lower for keyword in keywords):
                return category

        return "other"

    def analyze_themes_with_llm(
        self,
        reviews: List[Dict],
        product_type: str,
        metrics: Dict,
        previous_response_id: str = None,
        return_full_response: bool = False,
    ) -> Any:
        """Analyzes themes using the GPT-5 Responses API and context chaining."""
        logger.info("🎯 Analyzing themes with GPT-5 Mini...")

        # Include metrics context
        metrics_context = f"""
Product Metrics:
- Total Reviews: {metrics.get('total_reviews', 0)}
- Average Rating: {metrics.get('average_rating', 0):.1f}/5
- Sentiment Distribution: {metrics.get('sentiment_distribution', {})}
- Product Type: {product_type}
"""

        # A cleaner format for the input reviews
        review_lines = "\n".join([
            f"Review (Rating: {r.get('review_rating', 'N/A')}): {r['review_text']}" 
            for r in reviews
        ])

        prompt = f"""
<instructions>
You are a product analyst. Follow this process:
1.  Read all reviews and identify 5-8 major themes.
2.  For each theme, create a list of specific positive points, a separate list of negative points, a 'confidence' score, the overall 'sentiment', and select one representative 'example_quote'.
3.  Format the result as a single JSON object. The keys of this object MUST be the theme names (e.g., "Durability"). The value for each key MUST be another object containing the analysis.

{metrics_context}

Your entire response must be a single JSON object. DO NOT nest it inside another key.
</instructions>

Example of the required output format:
{{
  "Thumbstick Wear": {{
    "sentiment": "negative",
    "confidence": 0.9,
    "positives": [],
    "negatives": ["Thumbsticks grind against the friction rings", "Plastic shows gouges after a few hours"],
    "example_quote": "the thumb sticks grind against the terrible friction ring..."
  }}
}}

Reviews to Analyze:
{review_lines}
"""

        try:
            # Build the parameter dictionary for the API call
            params = {
                "model": "gpt-5",
                "input": prompt,
                "reasoning": {"effort": "low"},
            }
            # Add the previous response ID if it exists to chain the reasoning
            if previous_response_id:
                params["previous_response_id"] = previous_response_id

            response = self.openai_client.responses.create(**params)

            # Return the full response object if needed for further chaining
            if return_full_response:
                return response

            # Otherwise, parse and return the JSON content
            raw_output = response.output[0].content[0].text
            themes_data = json.loads(raw_output)

            # The prompt asks for a root key "themes", so we extract that.
            return themes_data.get("themes", themes_data)

        except Exception as e:
            logger.error(f"Theme analysis failed: {e}")
            return self.fallback_theme_analysis(reviews, product_type)

    def generate_insights_with_llm(
        self,
        themes: Dict,
        issues: List[Dict],
        metrics: Dict,
        product_type: str,
        previous_response_id: str = None,
    ) -> Dict:
        """Generates final insights using the GPT-5 Responses API and full reasoning context."""
        logger.info("💡 Generating insights with GPT-5...")

        context = {
            "product_type": product_type,
            "themes": themes,
            "issues": issues,
            "metrics": metrics,
        }

        prompt = """
    <instructions>
    You are a senior product strategist. Based on the following analysis data, please do the following:

    Step 1: Identify key patterns, challenges, or opportunities from the data, paying close attention to the specific `positives` and `negatives` listed within each theme, as well as the overall metrics and rating distribution.
    Step 2: Write a clear and concise 2–3 sentence executive summary. Crucially, use the specific details from your analysis instead of generic terms (e.g., mention 'thumbstick grinding' instead of 'durability concerns').
    Step 3: Generate 3 to 5 actionable product or business recommendations that directly address the specific `negatives` you found. For each recommendation, provide the recommendation, priority, impact, and rationale.
    Step 4: Summarize 3 to 6 additional key insights or observations based on the specific `positives` and `negatives` and the sentiment/rating distribution.

    Return the result using this JSON format ONLY:
    {
    "executive_summary": "",
    "recommendations": [...],
    "key_insights": ["..."]
    }
    </instructions>

    Analysis Data:
    """ + json.dumps(
            context
        )

        try:
            # Build the parameter dictionary for the API call
            params = {
                "model": "gpt-5",
                "input": prompt,
                "reasoning": {
                    "effort": "medium"
                },  # Using the default for this complex task
            }
            # Add the previous response ID to complete the reasoning chain
            if previous_response_id:
                params["previous_response_id"] = previous_response_id

            response = self.openai_client.responses.create(**params)

            return self._parse_json_from_response(response)

        except Exception as e:
            logger.error(f"Insights generation failed: {e}")
            return {
                "executive_summary": "Analysis failed to generate.",
                "recommendations": [],
                "key_insights": [],
            }

    def calculate_enhanced_metrics(self, reviews: List[Dict], sentiments: List[Dict]) -> Dict[str, Any]:
        """Calculates aggregate metrics from the parsed review and sentiment data."""
        ratings = [r.get('review_rating') for r in reviews if r.get('review_rating') is not None]
        review_texts = [r.get('review_text', '') for r in reviews]
        
        if not reviews:
            return {
                'total_reviews': 0,
                'average_rating': 0,
                'sentiment_distribution': {},
                'rating_distribution': {},
                'average_review_length': 0,
                'analysis_quality': 'low',
            }

        # Calculate rating distribution
        rating_distribution = Counter(ratings) if ratings else {}
        
        # Calculate average review length
        avg_length = sum(len(text) for text in review_texts) / len(review_texts) if review_texts else 0
        
        # Determine analysis quality based on available data
        quality = 'high' if len(ratings) > len(reviews) * 0.8 else 'medium' if len(ratings) > len(reviews) * 0.5 else 'low'

        return {
            'total_reviews': len(reviews),
            'average_rating': round(sum(ratings) / len(ratings), 2) if ratings else 0,
            'sentiment_distribution': dict(Counter(s['sentiment'] for s in sentiments)),
            'rating_distribution': dict(rating_distribution),
            'average_review_length': round(avg_length, 0),
            'analysis_quality': quality,
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
            (r"(uncomfortable|hurt|pain)", "comfort"),
        ]

        issues = []
        for review in reviews:
            text = review["review_text"].lower()
            for pattern, issue_type in issue_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    if isinstance(match, tuple):
                        issue_description = f"{match[0]} {match[1]}"
                    else:
                        issue_description = match

                    issues.append(
                        {
                            "issue_name": issue_description,
                            "description": f"Pattern match: {issue_description}",
                            "frequency": 1,
                            "severity": "medium",
                            "example_quote": text[:100] + "...",
                            "type": issue_type,
                            "review_count": len(reviews),
                            "related_quotes": [text[:100] + "..."],
                        }
                    )

        # Aggregate duplicates
        issue_counts = Counter(issue["issue_name"] for issue in issues)
        unique_issues = []
        for issue in issues:
            if issue["issue_name"] not in [ui["issue_name"] for ui in unique_issues]:
                issue["frequency"] = issue_counts[issue["issue_name"]]
                unique_issues.append(issue)

        return sorted(unique_issues, key=lambda x: x["frequency"], reverse=True)[:5]

    def fallback_theme_analysis(
        self, reviews: List[Dict], product_type: str
    ) -> Dict[str, Any]:
        """Enhanced fallback theme analysis."""
        logger.info("Using enhanced fallback theme analysis")

        # Default themes by product type
        default_themes = {
            "gaming controller": [
                "build_quality",
                "battery_life",
                "connectivity",
                "comfort",
                "performance",
            ],
            "headphones": [
                "sound_quality",
                "comfort",
                "build_quality",
                "battery_life",
                "connectivity",
            ],
            "product": ["quality", "performance", "design", "price", "durability"],
        }

        themes = default_themes.get(product_type.lower(), default_themes["product"])

        theme_analysis = {}
        for theme in themes:
            theme_analysis[theme] = {
                "mentions": len(reviews) // len(themes),  # Rough estimate
                "sentiment": "neutral",
                "confidence": 0.5,
                "key_phrases": [theme.replace("_", " ")],
                "example_quote": f'Analysis of {theme.replace("_", " ")}',
            }

        return theme_analysis

    def generate_comprehensive_analysis(
        self, reviews: List[Dict], product_type: str = "product"
    ) -> Dict[str, Any]:
        logger.info(f"🚀 Starting GPT-5 chained analysis for {len(reviews)} reviews...")
        start_time = time.time()

        # Step 1: Rating-based sentiment analysis (no API call)
        sentiments = self.batch_sentiment_analysis(reviews)

        # Step 2: Calculate metrics early so they can be used in subsequent prompts
        metrics = self.calculate_enhanced_metrics(reviews, sentiments)

        # Step 3: Extract issues with metrics context
        issue_response = self.extract_issues_with_llm(
            reviews, product_type, metrics, return_full_response=True
        )
        issues = (
            self._parse_json_from_response(issue_response) if issue_response else []
        )

        previous_id = issue_response.id if issue_response else None

        # Step 4: Analyze themes with metrics context
        theme_response = self.analyze_themes_with_llm(
            reviews,
            product_type,
            metrics,
            previous_response_id=previous_id,
            return_full_response=True,
        )
        themes = (
            self._parse_json_from_response(theme_response) if theme_response else {}
        )

        previous_id = theme_response.id if theme_response else None

        # Step 5: Generate insights with full context
        insights = self.generate_insights_with_llm(
            themes, issues, metrics, product_type, previous_response_id=previous_id
        )

        analysis_time = time.time() - start_time

        return {
            "themes": themes,
            "issues": issues,
            "metrics": metrics,
            "insights": insights,
            "analysis_metadata": {
                "total_reviews": len(reviews),
                "analysis_date": datetime.now().isoformat(),
                "product_type": product_type,
                "model_used": "gpt-5-suite",
                "analysis_time_seconds": round(analysis_time, 2),
                "token_usage": self.token_usage.copy(),
            },
        }

    def create_detailed_report(self, analysis_results: Dict[str, Any]) -> str:
        """Create a detailed, actionable report in plain text."""

        themes = analysis_results["themes"]
        issues = analysis_results["issues"]
        metrics = analysis_results["metrics"]
        insights = analysis_results["insights"]
        metadata = analysis_results["analysis_metadata"]

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
        lines.append(insights.get("executive_summary", "Analysis completed."))
        lines.append("")

        # Key Insights
        if insights.get("key_insights"):
            lines.append("KEY INSIGHTS")
            lines.append("-" * 40)
            for insight in insights["key_insights"]:
                lines.append(f"- {insight}")
            lines.append("")

        # Key Metrics
        lines.append("KEY METRICS")
        lines.append("-" * 40)
        lines.append(f"Average Rating: {metrics['average_rating']:.1f}/5")
        lines.append(f"Total Reviews: {metrics['total_reviews']}")
        lines.append(
            f"Average Review Length: {metrics['average_review_length']:.0f} characters"
        )
        lines.append(f"Analysis Quality: {metrics['analysis_quality'].title()}")
        lines.append("")

        # Rating Distribution
        lines.append("Rating Distribution:")
        for rating in sorted(metrics["rating_distribution"].keys(), reverse=True):
            count = metrics["rating_distribution"][rating]
            percentage = (count / metrics['total_reviews'] * 100) if metrics['total_reviews'] > 0 else 0
            lines.append(f"  {rating} stars: {count} ({percentage:.1f}%)")
        lines.append("")

        # Sentiment Distribution
        lines.append("Sentiment Distribution:")
        total_sentiment = sum(metrics["sentiment_distribution"].values())
        for sentiment, count in metrics["sentiment_distribution"].items():
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
                lines.append(
                    f"   Frequency: {issue.get('frequency', 'N/A')} mention(s)"
                )
                lines.append(f"   Severity: {issue['severity'].title()}")
                lines.append(f"   Example: \"{issue['example_quote']}\"")
                lines.append("")
        else:
            lines.append("No significant issues identified.")
            lines.append("")

        # Theme Analysis
        lines.append("THEME ANALYSIS")
        lines.append("-" * 40)

        if themes:
            for theme_name, theme_data in themes.items():
                lines.append(f"{theme_name.replace('_', ' ').title()}")
                lines.append(f"   Mentions: {theme_data.get('mentions', 'N/A')}")
                lines.append(f"   Sentiment: {theme_data.get('sentiment', 'N/A').title()}")
                lines.append(f"   Confidence: {theme_data.get('confidence', 0):.2f}")

                if theme_data.get("positives"):
                    lines.append(f"   Positives: {', '.join(theme_data['positives'])}")
                
                if theme_data.get("negatives"):
                    lines.append(f"   Negatives: {', '.join(theme_data['negatives'])}")

                if theme_data.get("example_quote"):
                    lines.append(f"   Example: \"{theme_data['example_quote']}\"")
                lines.append("")

        # Recommendations
        lines.append("AI-POWERED RECOMMENDATIONS")
        lines.append("-" * 40)

        if insights.get("recommendations"):
            for i, rec in enumerate(insights["recommendations"], 1):
                lines.append(f"{i}. {rec['recommendation']}")
                lines.append(f"   Priority: {rec['priority'].title()}")
                lines.append(f"   Impact: {rec['impact']}")
                lines.append(f"   Rationale: {rec['rationale']}")
                lines.append("")

        # Analysis Statistics
        lines.append("ANALYSIS STATISTICS")
        lines.append("-" * 40)
        lines.append(f"Tokens Used: {metadata['token_usage']['total_tokens']:,}")
        lines.append(
            f"Estimated Cost: ${metadata['token_usage']['estimated_cost']:.4f}"
        )
        lines.append(f"Processing Time: {metadata['analysis_time_seconds']}s")
        lines.append("")

        lines.append("=" * 80)
        lines.append("Analysis Complete")
        lines.append("=" * 80)

        return "\n".join(lines)


# Flask app setup
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["*"]}}, supports_credentials=False)


@app.route("/api/analyze-html", methods=["POST"])
def analyze_html():
    """API endpoint for analyzing HTML content."""
    try:
        # Get HTML from request
        data = request.get_json(force=True)
        html_content = data.get("html", "")
        product_type = data.get("product_type", "product")

        if not html_content:
            return jsonify({"error": "No HTML content provided"}), 400

        # Use the comprehensive parsing function
        reviews = parse_reviews_comprehensive(html_content)

        if not reviews:
            return jsonify({"error": "No reviews found in HTML"}), 400

        # Limit reviews for analysis
        reviews = reviews[:ANALYSIS_REVIEW_LIMIT]

        # Initialize analyzer
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            return jsonify({"error": "OpenAI API key not configured"}), 500

        analyzer = OpenAIReviewAnalyzer(openai_api_key)

        # Generate comprehensive analysis
        analysis_results = analyzer.generate_comprehensive_analysis(
            reviews, product_type
        )

        # Return results in format expected by React component
        response = {
            "reviews": [r["review_text"] for r in reviews],
            "sentiment": analysis_results["metrics"]["sentiment_distribution"],
            "themes": analysis_results["themes"],
            "issues": analysis_results["issues"],
            "insights": analysis_results["insights"],
            "summary": analysis_results["insights"].get("executive_summary", ""),
            "analysis_metadata": {
                "total_reviews": analysis_results["metrics"]["total_reviews"],
                "analysis_date": datetime.now().isoformat(),
                "product_type": product_type,
                "model_used": analysis_results["analysis_metadata"]["model_used"],
                "analysis_time_seconds": analysis_results["analysis_metadata"][
                    "analysis_time_seconds"
                ],
                "token_usage": {
                    "total_tokens": analysis_results["analysis_metadata"][
                        "token_usage"
                    ]["total_tokens"],
                    "estimated_cost": round(
                        analysis_results["analysis_metadata"]["token_usage"][
                            "estimated_cost"
                        ],
                        4,
                    ),
                },
            },
        }

        # Memory cleanup
        gc.collect()

        return jsonify(response)

    except Exception as e:
        logger.error(f"API analysis failed: {e}", exc_info=True)
        return (
            jsonify({"error": "Analysis failed", "message": str(e), "success": False}),
            500,
        )


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "review-analyzer",
            "version": "3.0",
        }
    )


def main():
    """Main execution function for standalone usage."""
    try:
        # Get OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            openai_api_key = input("Enter your OpenAI API key: ").strip()

        if not openai_api_key:
            print("OpenAI API key is required.")
            return

        # Download required NLTK data
        try:
            nltk.download("punkt", quiet=True)
        except Exception:
            logger.warning("Could not download NLTK 'punkt' data.")
            pass

        # Check for sample HTML file
        html_file = "sample.html"
        if not os.path.exists(html_file):
            print(f"HTML file '{html_file}' not found.")
            return

        # Parse reviews using comprehensive function
        with open(html_file, "r", encoding="utf-8") as f:
            html_content = f.read()

        reviews = parse_reviews_comprehensive(html_content)
        if not reviews:
            print("No reviews found or error parsing file.")
            return

        print(f"Found {len(reviews)} reviews to analyze...")

        # Get product type
        product_type = input(
            "Enter product type (e.g., 'gaming controller', 'headphones', 'laptop'): "
        ).strip()
        if not product_type:
            product_type = "product"

        # Initialize analyzer
        print("Initializing AI analyzer...")
        analyzer = OpenAIReviewAnalyzer(openai_api_key)

        # Generate comprehensive analysis
        print("Running AI analysis...")
        analysis_results = analyzer.generate_comprehensive_analysis(
            reviews, product_type
        )

        # Generate and display report
        print("Generating detailed report...")
        report = analyzer.create_detailed_report(analysis_results)

        print("\n" + report)

        # Export results
        with open("analysis_results.json", "w", encoding="utf-8") as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False, default=str)

        print(f"\nAnalysis complete! Results exported to analysis_results.json")
        print(
            f"Total cost: ${analysis_results['analysis_metadata']['token_usage']['estimated_cost']:.4f}"
        )

    except Exception as e:
        logger.error(f"Standalone analysis failed: {e}")
        print(f"Analysis failed: {e}")


if __name__ == "__main__":
    # Check if running as Flask app or standalone
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "standalone":
        main()
    else:
        port = int(os.getenv("PORT", 5000))
        app.run(debug=True, port=port, host="0.0.0.0")