import gc
import json
import re
import os
import logging
import time
from datetime import datetime
from collections import Counter
from typing import List, Dict, Any, Optional

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

# Load configuration
env_key = 'OPENAI_API_KEY'
load_dotenv()
OPENAI_API_KEY = os.getenv(env_key)
if not OPENAI_API_KEY:
    raise RuntimeError(f"Environment variable '{env_key}' is required.")

# Constants for memory control
MAX_HTML_SIZE = 3 * 1024 * 1024  # 3MB
PARSE_HTML_LIMIT = 200
ANALYSIS_REVIEW_LIMIT = 50
BATCH_SIZE = 8
NEGATIVE_SAMPLE = 10

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
CORS(
    app,
    resources={r"/api/*": {"origins": r"^https://.*\.vercel\.app$"}},
    supports_credentials=True
)

class OpenAIReviewAnalyzer:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.token_usage = {'total': 0, 'cost': 0.0}
        logger.info("Analyzer initialized.")

    def _track_usage(self, usage):
        tokens = usage.total_tokens
        self.token_usage['total'] += tokens
        self.token_usage['cost'] += tokens * 0.00015 / 1000

    def batch_sentiment_analysis(self, reviews: List[Dict]) -> List[Dict]:
        sentiments = []
        for i in range(0, len(reviews), BATCH_SIZE):
            batch = reviews[i:i + BATCH_SIZE]
            lines = [f"{i+idx}|{r.get('review_rating', 'N/A')}|{r['review_text'][:200]}" \
                     for idx, r in enumerate(batch)]
            prompt = (
                "Analyze sentiment for these reviews (index|rating|text):\n" +
                "\n".join(lines) +
                "\nReturn JSON array: [{\"review_index\":0,\"sentiment\":\"...\",\"confidence\":0.0}]"
            )
            try:
                res = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=500
                )
                self._track_usage(res.usage)
                data = json.loads(res.choices[0].message.content)
            except Exception:
                # Fallback quick heuristic
                data = []
                for idx, r in enumerate(batch):
                    rating = r.get('review_rating', 3)
                    if rating >= 4:
                        data.append({'review_index': i+idx, 'sentiment': 'positive', 'confidence': 0.7})
                    elif rating <= 2:
                        data.append({'review_index': i+idx, 'sentiment': 'negative', 'confidence': 0.7})
                    else:
                        data.append({'review_index': i+idx, 'sentiment': 'neutral', 'confidence': 0.5})
            sentiments.extend(data)
            del batch, lines, data
            gc.collect()
        return sentiments

    def extract_issues_with_llm(self, reviews: List[Dict], product_type: str) -> List[Dict]:
        neg = [r for r in reviews[:ANALYSIS_REVIEW_LIMIT] if r.get('review_rating',3) <= 2][:NEGATIVE_SAMPLE]
        if not neg:
            neg = reviews[:5]
        prompt = "Find issues in these reviews:\n" + \
                 "\n".join(f"{i}: {r['review_text'][:250]}" for i, r in enumerate(neg)) + \
                 "\nReturn JSON: [{\"issue_name\":...}]"
        try:
            res = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800
            )
            self._track_usage(res.usage)
            issues = json.loads(res.choices[0].message.content)
        except Exception:
            issues = []
        # categorize and trim
        for issue in issues[:5]:
            issue['type'] = self.categorize(issue['issue_name'])
        return issues[:5]

    def categorize(self, name: str) -> str:
        lookup = {
            'connectivity': ['connect', 'wifi'],
            'quality': ['build', 'cheap'],
            'performance': ['slow', 'lag'],
            'battery': ['battery', 'power'],
            'durability': ['broken', 'fail'],
            'comfort': ['comfort'],
            'functionality': ['function', 'button']
        }
        lower = name.lower()
        for cat, keys in lookup.items():
            if any(k in lower for k in keys): return cat
        return 'other'

    def analyze_themes_with_llm(self, reviews: List[Dict], product_type: str) -> Dict[str, Any]:
        sample = reviews[:6]
        prompt = "Analyze themes for these reviews:\n" + \
                 "\n".join(f"{i}: {r['review_text'][:200]}" for i, r in enumerate(sample)) + \
                 "\nReturn JSON with discovered_themes and details."
        try:
            res = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800
            )
            self._track_usage(res.usage)
            themes = json.loads(res.choices[0].message.content)
        except Exception:
            themes = {'themes': {}, 'discovered_themes': [], 'sample_size': 0}
        themes['sample_size'] = len(sample)
        return themes

    def calculate_metrics(self, reviews: List[Dict], sentiments: List[Dict]) -> Dict[str, Any]:
        ratings = [r.get('review_rating',3) for r in reviews]
        lengths = [len(r['review_text']) for r in reviews]
        counts = Counter(s['sentiment'] for s in sentiments)
        conf = [s['confidence'] for s in sentiments]
        avg_conf = sum(conf)/len(conf) if conf else 0
        avg_len = sum(lengths)/len(lengths) if lengths else 0
        reading_ease = max(0, 100 - avg_len/10)
        return {
            'total_reviews': len(reviews),
            'average_rating': sum(ratings)/len(ratings) if ratings else 0,
            'rating_distribution': dict(Counter(ratings)),
            'avg_review_length': avg_len,
            'sentiment_distribution': dict(counts),
            'reading_ease': reading_ease,
            'avg_confidence': avg_conf,
            'analysis_quality': 'enhanced' if avg_conf>0.7 else 'standard'
        }

    def generate_insights_with_llm(self, themes: Dict, issues: List[Dict], metrics: Dict, product_type: str) -> Dict:
        context = json.dumps({'themes': themes, 'issues': issues, 'metrics': metrics}, indent=2)
        prompt = f"Based on context provide summary, recommendations, and impact as JSON.\n{context}"
        try:
            res = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            self._track_usage(res.usage)
            return json.loads(res.choices[0].message.content)
        except Exception:
            return {'executive_summary': '', 'recommendations': [], 'key_insights': []}

    def comprehensive_analysis(self, reviews: List[Dict], product_type: str) -> Dict[str, Any]:
        gc.collect()
        sentiments = self.batch_sentiment_analysis(reviews)
        issues = self.extract_issues_with_llm(reviews, product_type)
        themes = self.analyze_themes_with_llm(reviews, product_type)
        metrics = self.calculate_metrics(reviews, sentiments)
        insights = self.generate_insights_with_llm(themes, issues, metrics, product_type)
        return {
            'themes': themes,
            'issues': issues,
            'metrics': metrics,
            'insights': insights,
            'metadata': {
                'time': round(time.time(),2),
                'tokens': self.token_usage
            }
        }

# HTML parsing

def parse_reviews(html: str) -> List[Dict]:
    if len(html) > MAX_HTML_SIZE:
        html = html[:MAX_HTML_SIZE]
    try:
        soup = BeautifulSoup(html, 'lxml')
    except:
        soup = BeautifulSoup(html, 'html.parser')
    elems = soup.select_one('li.review-item, .review-item, .review, [data-testid="review"]')
    reviews = []
    for el in soup.select('li.review-item, .review-item, .review, [data-testid="review"]')[:PARSE_HTML_LIMIT]:
        text_el = el.select_one('p.pre-white-space, .review-text, p, .content')
        txt = text_el.get_text(strip=True)[:2000] if text_el else ''
        if not txt:
            continue
        rating = None
        rm = el.select_one('p.visually-hidden')
        if rm:
            m = re.search(r'Rated (\d(?:\.\d)?)', rm.text)
            if m: rating = float(m.group(1))
        reviews.append({'review_text': txt, 'review_rating': rating})
    del soup
    gc.collect()
    return reviews

@app.route('/api/analyze-html', methods=['POST'])
def analyze_html():
    data = request.get_json(force=True)
    html = data.get('html', '')
    if not html:
        return jsonify({'error': 'No HTML provided'}), 400
    reviews = parse_reviews(html)
    if not reviews:
        return jsonify({'error': 'No reviews parsed'}), 400
    if len(reviews) > ANALYSIS_REVIEW_LIMIT:
        reviews = reviews[:ANALYSIS_REVIEW_LIMIT]
    analyzer = OpenAIReviewAnalyzer(OPENAI_API_KEY)
    result = analyzer.comprehensive_analysis(reviews, data.get('product_type', 'product'))
    response = {
        'reviews': [r['review_text'][:500] for r in reviews[:20]],
        'sentiment': result['metrics']['sentiment_distribution'],
        'themes': result['themes'],
        'issues': result['issues'][:3],
        'insights': result['insights'],
        'summary': result['insights'].get('executive_summary', ''),
        'metadata': result['metadata']
    }
    del analyzer, reviews, result
    gc.collect()
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=int(os.getenv('PORT', 5000)))
