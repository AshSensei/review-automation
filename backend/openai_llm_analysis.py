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
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
CORS(
    app,
    resources={r"/api/*": {"origins": ["*"]}},  # Allow all origins for now
    supports_credentials=False  # Set to False when allowing all origins
)


class OpenAIReviewAnalyzer:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.token_usage = {"total": 0, "cost": 0.0}
        logger.info("Analyzer initialized.")

    def _track_usage(self, usage):
        tokens = usage.total_tokens
        self.token_usage["total"] += tokens
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
                "Analyze the sentiment of these reviews.\n"
                "Use context clues and tone.\n"
                "Format: index|rating|text.\n" +
                "\n".join(lines) +
                "\n\nReturn EXACT JSON:"
                "[{\"review_index\":0,\"sentiment\":\"positive|negative|neutral\",\"confidence\":0.8}]"
            )
            try:
                res = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=500,
                )
                self._track_usage(res.usage)
                content = re.sub(r"^```json|```$", "", res.choices[0].message.content.strip())
                data = json.loads(content)
            except Exception as e:
                logger.warning(f"Sentiment LLM failed: {e}")
                data = []
                for idx, r in enumerate(batch):
                    rating = r.get("review_rating")
                    if rating is None:
                        sentiment, conf = "neutral", 0.5
                    else:
                        sentiment = "positive" if rating >= 4 else "negative" if rating <= 2 else "neutral"
                        conf = 0.7 if sentiment != "neutral" else 0.5
                    data.append({
                        "review_index": i + idx,
                        "sentiment": sentiment,
                        "confidence": conf,
                    })
            sentiments.extend(data)
            gc.collect()
        return sentiments

    def extract_issues_with_llm(self, reviews: List[Dict], product_type: str) -> List[Dict]:
        neg = [
            r for r in reviews[:ANALYSIS_REVIEW_LIMIT]
            if r.get("review_rating", 3) <= 2
        ][:NEGATIVE_SAMPLE]
        if not neg:
            neg = reviews[:5]
        prompt = f"""
Analyze these {product_type} reviews and extract up to 5 concrete issues.
Return JSON EXACTLY:
[
  {{
    "issue_name": "unique_name",
    "description": "brief description",
    "severity": "high|medium|low",
    "frequency": number,
    "example_quote": "..."
  }},...
]
Reviews:
""" + "\n".join(f"Review {i+1}: {r['review_text'][:250]}" for i, r in enumerate(neg))
        try:
            res = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800,
            )
            self._track_usage(res.usage)
            content = re.sub(r"^```json|```$", "", res.choices[0].message.content.strip())
            issues = json.loads(content)
        except Exception as e:
            logger.warning(f"Issue extraction failed: {e}")
            issues = []
        enriched = []
        for issue in issues[:5]:
            name = issue.get("issue_name", "")
            enriched.append({
                "issue_name": name,
                "description": issue.get("description", ""),
                "severity": issue.get("severity", "medium"),
                "review_count": issue.get("frequency", len(neg)),
                "example_quote": issue.get("example_quote", ""),
                "related_quotes": [issue.get("example_quote", "")],
                "type": self._categorize_issue(name)
            })
        return enriched

    def analyze_themes_with_llm(self, reviews: List[Dict], product_type: str) -> Dict[str, Any]:
        sample = reviews[:min(len(reviews), 12)]
        prompt = f"""
Analyze these {product_type} reviews and identify 8-10 specific themes.
Return JSON EXACTLY:
{{
  "discovered_themes": ["theme1","theme2",...],
  "sample_size": {len(sample)},
  "themes": {{
    "theme1": {{
      "mentions": number,
      "sentiment": "positive|negative|mixed",
      "confidence": 0.0-1.0,
      "key_phrases": ["..."],
      "example_quote": "..."
    }},...
  }}
}}
Reviews:
""" + "\n".join(f"Review {i+1}: {r['review_text'][:200]}" for i, r in enumerate(sample))
        try:
            res = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1000,
            )
            self._track_usage(res.usage)
            content = re.sub(r"^```json|```$", "", res.choices[0].message.content.strip())
            themes = json.loads(content)
        except Exception as e:
            logger.warning(f"Theme analysis failed: {e}")
            themes = {"discovered_themes": [], "sample_size": len(sample), "themes": {}}
        return themes

    def generate_insights_with_llm(self, themes: Dict, issues: List[Dict], metrics: Dict, product_type: str) -> Dict[str, Any]:
        context = {"product_type": product_type, "themes": themes, "issues": issues, "metrics": metrics}
        prompt = """
Based on this analysis, provide:
1. executive_summary (2-3 sentences)
2. recommendations: array of { recommendation, priority, impact, rationale }
3. key_insights: array of strings
Return JSON EXACTLY:
{"executive_summary":"","recommendations":[{"recommendation":"","priority":"high|medium|low","impact":"","rationale":""}],"key_insights":["..."]}
""" + json.dumps(context)
        try:
            res = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000,
            )
            self._track_usage(res.usage)
            content = re.sub(r"^```json|```$", "", res.choices[0].message.content.strip())
            insights = json.loads(content)
        except Exception as e:
            logger.warning(f"Insights generation failed: {e}")
            insights = {"executive_summary": "", "recommendations": [], "key_insights": []}
        return insights

    def comprehensive_analysis(self, reviews: List[Dict], product_type: str) -> Dict[str, Any]:
        start_ts = time.time()
        sentiments = self.batch_sentiment_analysis(reviews)
        issues = self.extract_issues_with_llm(reviews, product_type)
        themes = self.analyze_themes_with_llm(reviews, product_type)
        metrics = self.calculate_metrics(reviews, sentiments)
        insights = self.generate_insights_with_llm(themes, issues, metrics, product_type)
        return {
            "themes": themes,
            "issues": issues,
            "metrics": metrics,
            "insights": insights,
            "metadata": {
                "timestamp": start_ts,
                "token_usage": self.token_usage,
                "processed_reviews": len(reviews)
            }
        }

@app.route("/api/analyze-html", methods=["POST"])
def analyze_html():
    try:
        data = request.get_json(force=True)
        html_input = data.get("html", "")
        product_type = data.get("product_type", "product")
        if not html_input:
            return jsonify({"error": "No HTML provided"}), 400
        reviews = parse_reviews(html_input)[:ANALYSIS_REVIEW_LIMIT]
        analyzer = OpenAIReviewAnalyzer(OPENAI_API_KEY)
        result = analyzer.comprehensive_analysis(reviews, product_type)
        response = {
            "reviews": [r["review_text"] for r in reviews],
            "sentiment": result["metrics"]["sentiment_distribution"],
            "themes": result["themes"],
            "issues": result["issues"],
            "insights": result["insights"],
            "summary": result["insights"].get("executive_summary", ""),
            "analysis_metadata": {
                "total_reviews": result["metrics"]["total_reviews"],
                "analysis_date": datetime.now().isoformat(),
                "product_type": product_type,
                "model_used": "gpt-4o-mini + local models",
                "analysis_time_seconds": round(time.time() - result["metadata"]["timestamp"], 2),
                "token_usage": {"total_tokens": result["metadata"]["token_usage"]["total"],
                    "estimated_cost": round(result["metadata"]["token_usage"]["cost"], 4)
                }
            }
        }
        gc.collect()
        return jsonify(response)
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return jsonify({"error": "Analysis failed", "message": str(e), "success": False}), 500

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat(), "service": "review-analyzer"})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(debug=True, port=port, host="0.0.0.0")
