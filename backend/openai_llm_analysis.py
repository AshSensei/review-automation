import gc
import json
import re
import os
import logging
import time
from datetime import datetime
from collections import Counter
from typing import List, Dict, Any, Optional

from io import StringIO
from xml.etree.ElementTree import iterparse, ParseError
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from bs4 import BeautifulSoup

# Load configuration
env_key = "OPENAI_API_KEY"
load_dotenv()
OPENAI_API_KEY = os.getenv(env_key)
if not OPENAI_API_KEY:
    raise RuntimeError(f"Environment variable '{env_key}' is required.")

# Memory-safety caps
MAX_HTML_SIZE = 3 * 1024 * 1024  # 3MB
PARSE_HTML_LIMIT = 30  # max reviews to parse
ANALYSIS_REVIEW_LIMIT = 30  # max reviews to analyze
BATCH_SIZE = 4  # reviews per LLM call
NEGATIVE_SAMPLE = 5  # negative reviews sample size

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
CORS(
    app,
    resources={r"/api/*": {"origins": r"^https://.*\.vercel\.app$"}},
    supports_credentials=False,
)


class OpenAIReviewAnalyzer:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.token_usage = {"total": 0, "cost": 0.0}
        logger.info("Analyzer initialized.")

    def _track_usage(self, usage):
        tokens = usage.total_tokens
        self.token_usage["total"] += tokens
        # Updated pricing for gpt-4o-mini (as of 2024)
        self.token_usage["cost"] += tokens * 0.000150 / 1000

    def batch_sentiment_analysis(self, reviews: List[Dict]) -> List[Dict]:
        sentiments = []
        for i in range(0, len(reviews), BATCH_SIZE):
            batch = reviews[i : i + BATCH_SIZE]
            lines = [
                f"{i+idx}|{r.get('review_rating','N/A')}|{r['review_text'][:200]}"
                for idx, r in enumerate(batch)
            ]
            prompt = (
                "Analyze the sentiment of these reviews (format: index|rating|text):\n"
                + "\n".join(lines)
                + "\n\nReturn only valid JSON in this exact format: "
                '[{"review_index":0,"sentiment":"positive|negative|neutral","confidence":0.8}]'
            )
            try:
                res = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=500,
                )
                self._track_usage(res.usage)
                content = res.choices[0].message.content.strip()
                # Clean up any markdown formatting
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                data = json.loads(content)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(
                    f"LLM sentiment analysis failed: {e}, falling back to rating-based"
                )
                data = []
                for idx, r in enumerate(batch):
                    rating = r.get("review_rating")
                    if rating is None:
                        sentiment = "neutral"
                        conf = 0.5
                    else:
                        sentiment = (
                            "positive"
                            if rating >= 4
                            else "negative" if rating <= 2 else "neutral"
                        )
                        conf = 0.7 if sentiment != "neutral" else 0.5
                    data.append(
                        {
                            "review_index": i + idx,
                            "sentiment": sentiment,
                            "confidence": conf,
                        }
                    )
            sentiments.extend(data)
            del batch, lines, data
            gc.collect()
        return sentiments

    def extract_issues_with_llm(
        self, reviews: List[Dict], product_type: str
    ) -> List[Dict]:
        # Filter negative reviews, handling None ratings safely
        neg = [
            r
            for r in reviews[:ANALYSIS_REVIEW_LIMIT]
            if r.get("review_rating") is not None and r.get("review_rating") <= 2
        ][:NEGATIVE_SAMPLE]
        if not neg:
            neg = reviews[:5]  # fallback to first 5 if no negative reviews

        prompt = (
            f"Analyze these {product_type} reviews to identify common issues and problems:\n\n"
            + "\n".join(
                f"Review {i+1}: {r['review_text'][:250]}" for i, r in enumerate(neg)
            )
            + "\n\nReturn only valid JSON in this format: "
            '[{"issue_name":"Brief issue name","description":"Short description","frequency":3,"severity":"high|medium|low"}]'
        )

        try:
            res = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800,
            )
            self._track_usage(res.usage)
            content = res.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            issues = json.loads(content)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"LLM issue extraction failed: {e}")
            issues = []

        # Add categorization and limit results
        for issue in issues[:5]:
            issue["type"] = self._categorize_issue(issue.get("issue_name", ""))
        return issues[:5]

    def _categorize_issue(self, name: str) -> str:
        lookup = {
            "connectivity": ["connect", "wifi", "bluetooth", "network"],
            "quality": ["build", "cheap", "flimsy", "poor quality"],
            "performance": ["slow", "lag", "speed", "performance"],
            "battery": ["battery", "power", "charge", "drain"],
            "durability": ["broken", "fail", "break", "crack"],
            "comfort": ["comfort", "fit", "ergonomic"],
            "functionality": ["function", "button", "feature", "work"],
        }
        lower = name.lower()
        for cat, keys in lookup.items():
            if any(k in lower for k in keys):
                return cat
        return "other"

    def analyze_themes_with_llm(
        self, reviews: List[Dict], product_type: str
    ) -> Dict[str, Any]:
        sample = reviews[:6]
        prompt = (
            f"Analyze themes and topics in these {product_type} reviews:\n\n"
            + "\n".join(
                f"Review {i+1}: {r['review_text'][:200]}" for i, r in enumerate(sample)
            )
            + "\n\nReturn only valid JSON with themes, their frequencies, and sample quotes: "
            '{"themes":{"theme_name":"frequency"},"discovered_themes":["theme1","theme2"],"sample_quotes":["quote1","quote2"]}'
        )

        try:
            res = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800,
            )
            self._track_usage(res.usage)
            content = res.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            themes = json.loads(content)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"LLM theme analysis failed: {e}")
            themes = {"themes": {}, "discovered_themes": [], "sample_quotes": []}

        themes["sample_size"] = len(sample)
        return themes

    def calculate_metrics(
        self, reviews: List[Dict], sentiments: List[Dict]
    ) -> Dict[str, Any]:
        ratings = [
            r.get("review_rating")
            for r in reviews
            if r.get("review_rating") is not None
        ]
        lengths = [len(r["review_text"]) for r in reviews]
        counts = Counter(s["sentiment"] for s in sentiments)
        confs = [s["confidence"] for s in sentiments]

        avg_conf = sum(confs) / len(confs) if confs else 0
        avg_len = sum(lengths) / len(lengths) if lengths else 0
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        reading_ease = max(0, 100 - avg_len / 10)

        return {
            "total_reviews": len(reviews),
            "average_rating": round(avg_rating, 2),
            "rating_distribution": dict(Counter(ratings)) if ratings else {},
            "avg_review_length": round(avg_len, 1),
            "sentiment_distribution": dict(counts),
            "reading_ease": round(reading_ease, 1),
            "avg_confidence": round(avg_conf, 2),
            "analysis_quality": "enhanced" if avg_conf > 0.7 else "standard",
        }

    def generate_insights_with_llm(
        self, themes: Dict, issues: List[Dict], metrics: Dict, product_type: str
    ) -> Dict:
        context = {
            "product_type": product_type,
            "themes": themes,
            "issues": issues,
            "metrics": metrics,
        }

        prompt = (
            f"Based on this analysis of {product_type} reviews, provide insights:\n\n"
            + json.dumps(context, indent=2)
            + "\n\nReturn only valid JSON with: "
            '{"executive_summary":"Brief summary","recommendations":["rec1","rec2"],"key_insights":["insight1","insight2"]}'
        )

        try:
            res = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000,
            )
            self._track_usage(res.usage)
            content = res.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            return json.loads(content)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"LLM insights generation failed: {e}")
            return {
                "executive_summary": "Analysis completed with basic metrics.",
                "recommendations": ["Consider reviewing customer feedback patterns."],
                "key_insights": ["Review data processed successfully."],
            }

    def comprehensive_analysis(
        self, reviews: List[Dict], product_type: str
    ) -> Dict[str, Any]:
        logger.info(f"Starting comprehensive analysis of {len(reviews)} reviews")
        gc.collect()

        sentiments = self.batch_sentiment_analysis(reviews)
        issues = self.extract_issues_with_llm(reviews, product_type)
        themes = self.analyze_themes_with_llm(reviews, product_type)
        metrics = self.calculate_metrics(reviews, sentiments)
        insights = self.generate_insights_with_llm(
            themes, issues, metrics, product_type
        )

        return {
            "themes": themes,
            "issues": issues,
            "metrics": metrics,
            "insights": insights,
            "metadata": {
                "timestamp": round(time.time(), 2),
                "token_usage": self.token_usage,
                "processed_reviews": len(reviews),
            },
        }


def parse_reviews_bs4(html: str) -> List[Dict]:
    """Parse reviews using BeautifulSoup as fallback"""
    try:
        soup = BeautifulSoup(html[:MAX_HTML_SIZE], "html.parser")
        reviews = []

        # Look for common review patterns
        review_elements = soup.find_all(
            ["li", "div"], class_=re.compile(r"review|comment|feedback", re.I)
        )

        for elem in review_elements[:PARSE_HTML_LIMIT]:
            # Extract review text
            text_elem = elem.find(["p", "span", "div"], string=re.compile(r".{20,}"))
            if not text_elem:
                text_elem = elem.find(string=re.compile(r".{20,}"))

            if text_elem:
                text = (
                    text_elem.get_text()
                    if hasattr(text_elem, "get_text")
                    else str(text_elem)
                )
                text = text.strip()[:2000]

                if len(text) > 20:  # Minimum review length
                    # Extract rating
                    rating = None
                    rating_elem = elem.find(
                        string=re.compile(r"Rated (\d(?:\.\d)?)", re.I)
                    )
                    if rating_elem:
                        match = re.search(r"(\d(?:\.\d)?)", rating_elem)
                        if match:
                            rating = float(match.group(1))

                    reviews.append({"review_text": text, "review_rating": rating})

        return reviews
    except Exception as e:
        logger.error(f"BeautifulSoup parsing failed: {e}")
        return []


def parse_reviews_xml(html: str) -> List[Dict]:
    """Parse reviews using XML parser with proper error handling"""
    try:
        # Truncate and wrap in a root element to ensure valid XML
        raw = f"<root>{html[:MAX_HTML_SIZE]}</root>"
        stream = StringIO(raw)
        reviews = []
        count = 0

        # Use iterparse without the 'tag' parameter
        context_iter = iterparse(stream, events=("end",))

        for event, elem in context_iter:
            if count >= PARSE_HTML_LIMIT:
                break

            # Only process 'li' elements
            if elem.tag == "li":
                # Extract review text
                text_elem = elem.find(".//p")
                txt = (
                    text_elem.text.strip()[:2000]
                    if text_elem is not None and text_elem.text
                    else ""
                )

                # Extract rating
                rating = None
                rating_elem = elem.find('.//p[@class="visually-hidden"]')
                if rating_elem is not None and rating_elem.text:
                    match = re.search(r"Rated (\d(?:\.\d)?)", rating_elem.text)
                    rating = float(match.group(1)) if match else None

                if txt and len(txt) > 20:
                    reviews.append({"review_text": txt, "review_rating": rating})
                    count += 1

            # Clear the element to free memory
            elem.clear()

        del raw, stream, context_iter
        gc.collect()
        return reviews

    except (ParseError, Exception) as e:
        logger.warning(f"XML parsing failed: {e}, falling back to BeautifulSoup")
        return []


def parse_reviews(html: str) -> List[Dict]:
    """Parse reviews with multiple fallback strategies"""
    # Try XML parsing first
    reviews = parse_reviews_xml(html)

    # Fall back to BeautifulSoup if XML parsing fails
    if not reviews:
        reviews = parse_reviews_bs4(html)

    # If still no reviews, try simple regex extraction as last resort
    if not reviews:
        logger.info("Attempting regex-based extraction as last resort")
        text_matches = re.findall(r"<p[^>]*>([^<]{50,500})</p>", html[:MAX_HTML_SIZE])
        for i, text in enumerate(text_matches[:PARSE_HTML_LIMIT]):
            reviews.append({"review_text": text.strip(), "review_rating": None})

    logger.info(f"Parsed {len(reviews)} reviews")
    return reviews


@app.route("/api/analyze-html", methods=["POST"])
def analyze_html():
    try:
        data = request.get_json(force=True)
        html_input = data.get("html", "")
        product_type = data.get("product_type", "product")

        if not html_input:
            return jsonify({"error": "No HTML provided"}), 400

        logger.info(f"Received HTML of length: {len(html_input)}")

        # Parse reviews
        reviews = parse_reviews(html_input)
        if not reviews:
            return jsonify({"error": "No reviews could be parsed from the HTML"}), 400

        logger.info(f"Parsed {len(reviews)} reviews")

        # Limit reviews for analysis
        if len(reviews) > ANALYSIS_REVIEW_LIMIT:
            reviews = reviews[:ANALYSIS_REVIEW_LIMIT]

        # Analyze reviews
        analyzer = OpenAIReviewAnalyzer(OPENAI_API_KEY)
        result = analyzer.comprehensive_analysis(reviews, product_type)

        # Prepare response
        response = {
            "reviews": [r["review_text"] for r in reviews],
            "sentiment": result["metrics"]["sentiment_distribution"],
            "themes": result["themes"],
            "issues": result["issues"][:5],
            "insights": result["insights"],
            "summary": result["insights"].get("executive_summary", ""),
            "analysis_metadata": {
                "total_reviews": result["metrics"]["total_reviews"],
                "analysis_date": datetime.now().isoformat(),
                "product_type": product_type,
                "model_used": "gpt-4o-mini + local modDels",
                "analysis_time_seconds": round(
                    time.time() - result["metadata"]["timestamp"], 2
                ),
                "token_usage": {
                    "total_tokens": result["metadata"]["token_usage"]["total"],
                    "estimated_cost": round(
                        result["metadata"]["token_usage"]["cost"], 4
                    ),
                },
            },
        }

        # Cleanup
        del analyzer, reviews, result, data, html_input
        gc.collect()

        return jsonify(response)

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        return (
            jsonify({"error": "Analysis failed", "message": str(e), "success": False}),
            500,
        )


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "review-analyzer",
        }
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(debug=True, port=port, host="0.0.0.0")
