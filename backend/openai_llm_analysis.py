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
from xml.etree.ElementTree import iterparse
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

# Memory-safety caps
MAX_HTML_SIZE = 3 * 1024 * 1024  # 3MB
PARSE_HTML_LIMIT = 30             # max reviews to parse
ANALYSIS_REVIEW_LIMIT = 30       # max reviews to analyze
BATCH_SIZE = 4                   # reviews per LLM call
NEGATIVE_SAMPLE = 5              # negative reviews sample size

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
            lines = [f"{i+idx}|{r.get('review_rating','N/A')}|{r['review_text'][:200]}" \
                     for idx, r in enumerate(batch)]
            prompt = ("Analyze sentiment (index|rating|text):\n" + "\n".join(lines) +
                      "\nReturn JSON: [{\"review_index\":0,\"sentiment\":\"...\",\"confidence\":0.0}]")
            try:
                res = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"user","content":prompt}],
                    temperature=0.1,
                    max_tokens=500
                )
                self._track_usage(res.usage)
                data = json.loads(res.choices[0].message.content)
            except Exception:
                data = []
                for idx, r in enumerate(batch):
                    rating = r.get('review_rating',3)
                    sentiment = 'positive' if rating>=4 else 'negative' if rating<=2 else 'neutral'
                    conf = 0.7 if sentiment!='neutral' else 0.5
                    data.append({'review_index':i+idx,'sentiment':sentiment,'confidence':conf})
            sentiments.extend(data)
            del batch, lines, data
            gc.collect()
        return sentiments

    def extract_issues_with_llm(self, reviews: List[Dict], product_type: str) -> List[Dict]:
        neg = [r for r in reviews[:ANALYSIS_REVIEW_LIMIT] if r.get('review_rating',3)<=2][:NEGATIVE_SAMPLE]
        if not neg: neg = reviews[:5]
        prompt = "Find issues:\n" + "\n".join(f"{i}:{r['review_text'][:250]}" for i,r in enumerate(neg)) + "\nReturn JSON"
        try:
            res = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                temperature=0.1,
                max_tokens=800
            )
            self._track_usage(res.usage)
            issues = json.loads(res.choices[0].message.content)
        except Exception:
            issues = []
        for issue in issues[:5]: issue['type'] = self._categorize(issue['issue_name'])
        return issues[:5]

    def _categorize(self, name:str) -> str:
        lookup={'connectivity':['connect','wifi'],'quality':['build','cheap'],
                'performance':['slow','lag'],'battery':['battery','power'],
                'durability':['broken','fail'],'comfort':['comfort'],
                'functionality':['function','button']}
        lower=name.lower()
        for cat,keys in lookup.items():
            if any(k in lower for k in keys): return cat
        return 'other'

    def analyze_themes_with_llm(self, reviews: List[Dict], product_type: str) -> Dict[str,Any]:
        sample = reviews[:6]
        prompt = "Analyze themes:\n" + "\n".join(f"{i}:{r['review_text'][:200]}" for i,r in enumerate(sample)) + "\nReturn JSON"
        try:
            res = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                temperature=0.2,
                max_tokens=800
            )
            self._track_usage(res.usage)
            themes = json.loads(res.choices[0].message.content)
        except Exception:
            themes={'themes':{},'discovered_themes':[],'sample_size':0}
        themes['sample_size']=len(sample)
        return themes

    def calculate_metrics(self, reviews: List[Dict], sentiments: List[Dict]) -> Dict[str,Any]:
        ratings=[r.get('review_rating',3) for r in reviews]
        lengths=[len(r['review_text']) for r in reviews]
        counts=Counter(s['sentiment'] for s in sentiments)
        confs=[s['confidence'] for s in sentiments]
        avg_conf=sum(confs)/len(confs) if confs else 0
        avg_len=sum(lengths)/len(lengths) if lengths else 0
        reading_ease=max(0,100-avg_len/10)
        return {'total_reviews':len(reviews),'average_rating':sum(ratings)/len(ratings) if ratings else 0,
                'rating_distribution':dict(Counter(ratings)),'avg_review_length':avg_len,
                'sentiment_distribution':dict(counts),'reading_ease':reading_ease,
                'avg_confidence':avg_conf,'analysis_quality':'enhanced' if avg_conf>0.7 else 'standard'}

    def generate_insights_with_llm(self, themes:Dict, issues:List[Dict], metrics:Dict, product_type:str) -> Dict:
        context=json.dumps({'themes':themes,'issues':issues,'metrics':metrics})
        prompt=f"Based on context provide summary, recommendations, impact as JSON.\n{context}"
        try:
            res=self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            self._track_usage(res.usage)
            return json.loads(res.choices[0].message.content)
        except Exception:
            return {'executive_summary':'','recommendations':[],'key_insights':[]}

    def comprehensive_analysis(self, reviews:List[Dict], product_type:str) -> Dict[str,Any]:
        gc.collect()
        sentiments=self.batch_sentiment_analysis(reviews)
        issues=self.extract_issues_with_llm(reviews,product_type)
        themes=self.analyze_themes_with_llm(reviews,product_type)
        metrics=self.calculate_metrics(reviews,sentiments)
        insights=self.generate_insights_with_llm(themes,issues,metrics,product_type)
        return {'themes':themes,'issues':issues,'metrics':metrics,'insights':insights,
                'metadata':{'time':round(time.time(),2),'tokens':self.token_usage}}

# HTML parsing with streaming
def parse_reviews(html: str) -> List[Dict]:
    # Truncate raw HTML and release it after wrapping in stream
    raw = html[:MAX_HTML_SIZE]
    del html
    gc.collect()
    reviews=[]
    stream=StringIO(raw)
    context_iter=iterparse(stream, events=('end',), tag='li')
    count=0
    for _, elem in context_iter:
        if count>=PARSE_HTML_LIMIT: break
        text_el=elem.find('.//p')
        txt=text_el.text.strip()[:2000] if text_el is not None else ''
        rating=None
        rm=elem.find('.//p[@class="visually-hidden"]')
        if rm is not None:
            m=re.search(r'Rated (\d(?:\.\d)?)',rm.text)
            rating=float(m.group(1)) if m else None
        if txt:
            reviews.append({'review_text':txt,'review_rating':rating})
            count+=1
        elem.clear()
    del raw,stream,context_iter
    gc.collect()
    return reviews

@app.route('/api/analyze-html', methods=['POST'])
def analyze_html():
    data=request.get_json(force=True)
    html_input=data.get('html','')
    if not html_input:
        return jsonify({'error':'No HTML provided'}),400
    reviews=parse_reviews(html_input)
    if not reviews:
        return jsonify({'error':'No reviews parsed'}),400
    if len(reviews)>ANALYSIS_REVIEW_LIMIT:
        reviews=reviews[:ANALYSIS_REVIEW_LIMIT]
    analyzer=OpenAIReviewAnalyzer(OPENAI_API_KEY)
    result=analyzer.comprehensive_analysis(reviews,data.get('product_type','product'))
    response={
        'reviews':[r['review_text'][:500] for r in reviews[:20]],
        'sentiment':result['metrics']['sentiment_distribution'],
        'themes':result['themes'],
        'issues':result['issues'][:3],
        'insights':result['insights'],
        'summary':result['insights'].get('executive_summary',''),
        'metadata':result['metadata']
    }
    del analyzer,reviews,result,data,html_input
    gc.collect()
    return jsonify(response)

if __name__=='__main__':
    port=int(os.getenv('PORT',5000))
    app.run(debug=True,port=port)
