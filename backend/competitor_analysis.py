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

# --- Configuration and Setup ---

# Load environment variables from .env file
load_dotenv()

# Configure logging to provide informative output
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Memory-safety caps
MAX_HTML_SIZE = 3 * 1024 * 1024 * 5  # 3MB
PARSE_HTML_LIMIT = 200  # max reviews to parse
ANALYSIS_REVIEW_LIMIT = 200  # max reviews to analyze
MAX_ISSUES_TO_ENRICH = 5
# Flask app setup
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["*"]}}, supports_credentials=False)


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
            ".review-item",
            '[data-testid="review"]',
            ".review",
            ".customer-review",
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
                    text_element = element.select_one(
                        ".review-text, .review-content p, .ugc-review-body"
                    )

                review_text = text_element.get_text(strip=True) if text_element else ""

                if not review_text:
                    review_text = element.get_text(strip=True)

                review_data["review_text"] = review_text[:1000]

                rating = None
                rating_element = element.select_one(
                    'i[data-hook="review-star-rating"] span.a-icon-alt'
                )
                if rating_element:
                    rating_text = rating_element.get_text(strip=True)
                    rating_match = re.search(r"(\d+(\.\d)?)", rating_text)
                    if rating_match:
                        rating = float(rating_match.group(1))

                if rating is None:
                    generic_rating_selectors = [
                        '[aria-label*="star"]',
                        ".review-rating",
                        ".rating",
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


# --- Core Analysis Class ---


class OpenAIHybridReviewAnalyzer:
    """
    A sophisticated review analyzer that uses a hybrid approach:
    - Local transformer models for fast, bulk sentiment analysis.
    - OpenAI's GPT models for nuanced theme extraction, issue identification, and strategic insights.
    """

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
        self,
        reviews: List[Dict],
        product_type: str,
        metrics: Dict,
        return_full_response: bool = False,
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
            f"Review (Rating: {r.get('review_rating', 'N/A')}): {r['review_text']}"
            for r in negative_reviews
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
                        "frequency": issue.get("frequency", 1),
                        "example_quote": issue.get("example_quote", ""),
                        "related_quotes": [issue.get("example_quote", "")],
                        "type": self.categorize_issue(name),
                    }
                )
            return enriched_issues
        except Exception as e:
            logger.error(f"LLM issue extraction failed: {e}")
            return [] if not return_full_response else None

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
        reviews,
        product_type,
        metrics,
        previous_response_id=None,
        return_full_response=False,
    ):

        prompt = f"""
    <instructions>
    You are a product analyst. Your task is to analyze the provided customer reviews for a '{product_type}'.

    Step 1: Identify the top 5-7 most prominent themes discussed in the reviews. A theme is a general topic like "Battery Life" or "Build Quality".
    
    Step 2: For each theme you identified, provide the following details:
    - A list of specific positive points.
    - A list of specific negative points.
    - An overall sentiment for the theme ("positive", "negative", or "mixed").
    - A confidence score (0.0-1.0).
    - One representative quote from the reviews that best summarizes the theme.

    Your final output must be a single, valid JSON object where the keys are the theme names you identified.
    </instructions>

    Product: {product_type}
    Reviews: {chr(10).join([f"Review {i+1}: {r['review_text']}" for i, r in enumerate(reviews)])}
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
            themes_data = self._parse_json_from_response(response)

            # The prompt asks for a root key "themes", so we extract that.
            return themes_data.get("themes", themes_data) if themes_data else {}

        except Exception as e:
            logger.error(f"Theme analysis failed: {e}")
            return self.fallback_theme_analysis(reviews, product_type)

    def calculate_enhanced_metrics(
        self, reviews: List[Dict], sentiments: List[Dict]
    ) -> Dict[str, Any]:
        """Calculates aggregate metrics from the parsed review and sentiment data."""
        ratings = [
            r.get("review_rating")
            for r in reviews
            if r.get("review_rating") is not None
        ]
        review_texts = [r.get("review_text", "") for r in reviews]

        if not reviews:
            return {
                "total_reviews": 0,
                "average_rating": 0,
                "sentiment_distribution": {},
                "rating_distribution": {},
                "average_review_length": 0,
                "analysis_quality": "low",
            }

        # Calculate rating distribution
        rating_distribution = Counter(ratings) if ratings else {}

        # Calculate average review length
        avg_length = (
            sum(len(text) for text in review_texts) / len(review_texts)
            if review_texts
            else 0
        )

        # Determine analysis quality based on available data
        quality = (
            "high"
            if len(ratings) > len(reviews) * 0.8
            else "medium" if len(ratings) > len(reviews) * 0.5 else "low"
        )

        return {
            "total_reviews": len(reviews),
            "average_rating": round(sum(ratings) / len(ratings), 2) if ratings else 0,
            "sentiment_distribution": dict(Counter(s["sentiment"] for s in sentiments)),
            "rating_distribution": dict(rating_distribution),
            "average_review_length": round(avg_length, 0),
            "analysis_quality": quality,
        }

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

    def compare_products(
        self,
        analysis_a: Dict,
        analysis_b: Dict,
        previous_response_id: Optional[str] = None,
        return_full_response: bool = False,
    ) -> Dict[str, Any]:
        """Compare two product analyses and generate strategic insights using GPT-5."""
        name_a = analysis_a.get("analysis_metadata", {}).get(
            "product_type", "Product A"
        )
        name_b = analysis_b.get("analysis_metadata", {}).get(
            "product_type", "Product B"
        )
        logger.info(f"🔍 Comparing '{name_a}' vs '{name_b}'...")

        themes_dict_a = analysis_a.get("themes", {})
        themes_dict_b = analysis_b.get("themes", {})

        shared_themes = list(
            set(themes_dict_a.keys()).intersection(set(themes_dict_b.keys()))
        )

        # --- THIS IS THE NEW, ENRICHED CONTEXT ---
        # Build a detailed sentiment string to give the AI full context
        sentiment_details = []
        for theme in shared_themes:
            sentiment_a = themes_dict_a.get(theme, {}).get("overall_sentiment", "N/A")
            sentiment_b = themes_dict_b.get(theme, {}).get("overall_sentiment", "N/A")
            sentiment_details.append(
                f"- **{theme}**: {name_a} Sentiment = {sentiment_a}, {name_b} Sentiment = {sentiment_b}"
            )

        sentiment_summary_str = "\n".join(sentiment_details)

        comparison_context = f"""
        **Product A ({name_a}):**
        - Average Rating: {analysis_a['metrics'].get('average_rating', 'N/A')}
        - Top Issues: {[issue['issue_name'] for issue in analysis_a.get('issues', [])[:3]]}

        **Product B ({name_b}):**
        - Average Rating: {analysis_b['metrics'].get('average_rating', 'N/A')}
        - Top Issues: {[issue['issue_name'] for issue in analysis_b.get('issues', [])[:3]]}

        **Shared Theme Sentiments:**
        {sentiment_summary_str}
        """
        # --- END OF NEW CONTEXT SECTION ---

        prompt = f"""
        <instructions>
        You are a product strategy analyst. Your task is to compare two products based on the provided review analysis data.

        **Analysis Data:**
        {comparison_context}

        **Your Goal:**
        Generate a single, valid JSON object with two top-level keys:
        1. "summary_table": Create a markdown table that compares the sentiment for each shared theme. The columns must be: "Theme", "{name_a} Sentiment", "{name_b} Sentiment", and "Winner". Determine the winner based on the sentiment data provided in the "Shared Theme Sentiments" section.
        2. "strategic_insights": Create an object containing three keys:
            - "competitive_advantages": A list of strings detailing {name_a}'s key strengths over {name_b}.
            - "areas_to_improve": A list of strings identifying {name_a}'s primary weaknesses or disadvantages compared to {name_b}.
            - "recommendations": A list of actionable recommendation objects for {name_a}, each with "recommendation", "priority" ('high', 'medium', or 'low'), and "impact" keys.

        Your entire output must be only the JSON object.
        </instructions>
        """

        try:
            params = {
                "model": "gpt-5",
                "input": prompt,
                "reasoning": {"effort": "medium"},
            }
            if previous_response_id:
                params["previous_response_id"] = previous_response_id

            response = self.openai_client.responses.create(**params)
            self._track_usage(response.usage, "gpt-5")

            if return_full_response:
                return response

            comparison_insights = self._parse_json_from_response(response)
            if not comparison_insights:
                raise ValueError("Parsing comparison insights returned None.")

        except Exception as e:
            logger.error(f"LLM comparison generation failed: {e}")
            comparison_insights = {
                "summary_table": "Error generating comparison table.",
                "strategic_insights": {
                    "competitive_advantages": [],
                    "areas_to_improve": [],
                    "recommendations": [],
                },
            }

        return {
            "shared_themes": shared_themes,
            "unique_to_product_a": list(
                set(themes_dict_a.keys()) - set(themes_dict_b.keys())
            ),
            "unique_to_product_b": list(
                set(themes_dict_b.keys()) - set(themes_dict_a.keys())
            ),
            "theme_sentiment_comparison": self._create_theme_sentiment_comparison(
                analysis_a, analysis_b, shared_themes
            ),
            "summary_table": comparison_insights.get("summary_table", ""),
            "strategic_insights": comparison_insights.get("strategic_insights", {}),
        }

    def _create_theme_sentiment_comparison(self, analysis_a, analysis_b, shared_themes):
        """Helper to create a structured dict of theme sentiment comparison."""
        comparison = {}
        # Correctly access the themes dictionary with a single .get()
        themes_a = analysis_a.get("themes", {})
        themes_b = analysis_b.get("themes", {})

        for theme in shared_themes:
            comparison[theme] = {
                "product_a": themes_a.get(theme, {}).get(
                    "overall_sentiment", "neutral"
                ),
                "product_b": themes_b.get(theme, {}).get(
                    "overall_sentiment", "neutral"
                ),
            }
        return comparison


# --- Flask API Endpoints ---


@app.route("/api/compare", methods=["POST"])
def handle_compare_request():
    """API endpoint to compare two products based on their review HTML."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return jsonify({"error": "OpenAI API key not configured on the server."}), 500

    try:
        data = request.get_json(force=True)
        html_a = data.get("html_a")
        name_a = data.get("product_name_a", "Product A")
        html_b = data.get("html_b")
        name_b = data.get("product_name_b", "Product B")

        if not all([html_a, name_a, html_b, name_b]):
            return (
                jsonify(
                    {
                        "error": "Missing required fields: html_a, product_name_a, html_b, product_name_b"
                    }
                ),
                400,
            )

        # --- Analysis for Product A ---
        reviews_a = parse_reviews_comprehensive(html_a)
        if not reviews_a:
            return jsonify({"error": f"No reviews could be parsed for {name_a}."}), 400

        # --- Analysis for Product B ---
        reviews_b = parse_reviews_comprehensive(html_b)
        if not reviews_b:
            return jsonify({"error": f"No reviews could be parsed for {name_b}."}), 400

        # Initialize the analyzer once we know we have reviews
        analyzer = OpenAIHybridReviewAnalyzer(openai_api_key=api_key)

        # Generate comprehensive analysis for both
        analysis_a = analyzer.generate_comprehensive_analysis(reviews_a, name_a)
        analysis_b = analyzer.generate_comprehensive_analysis(reviews_b, name_b)

        # Compare the two analysis results
        comparison_result = analyzer.compare_products(analysis_a, analysis_b)

        # Assemble the final JSON output
        final_output = {
            "product_a": analysis_a,
            "product_b": analysis_b,
            "comparison": comparison_result,
        }

        gc.collect()  # Clean up memory
        return jsonify(final_output)

    except Exception as e:
        logger.error(f"An error occurred in /api/compare: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500


if __name__ == "__main__":
    # Runs the Flask app
    # Use a different port to avoid conflicts with other services
    app.run(debug=True, port=5000)
