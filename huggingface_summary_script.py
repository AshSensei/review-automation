import json
import re
from collections import Counter, defaultdict
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer
import nltk
import torch
from nltk.tokenize import sent_tokenize
import logging
from typing import List, Dict, Tuple, Optional
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductReviewAnalyzer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """Initialize the analyzer with configurable models."""
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.device = 0 if torch.cuda.is_available() else -1
        
        # Initialize models with error handling
        try:
            self.summarizer = pipeline(
                "summarization",
                model=self.model_name,
                tokenizer=self.tokenizer,
                device=self.device,
                framework="pt"
            )
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=self.device
            )
            logger.info(f"Models loaded successfully on device: {self.device}")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
        
        # Enhanced feature keywords with weights
        self.feature_keywords = {
            'build_quality': {
                'positive': ['durable', 'sturdy', 'solid', 'premium', 'well-built', 'robust'],
                'negative': ['cheap', 'flimsy', 'fragile', 'plastic', 'poorly-made', 'weak']
            },
            'ergonomics': {
                'positive': ['comfortable', 'ergonomic', 'natural-grip', 'fits-well'],
                'negative': ['uncomfortable', 'cramped', 'awkward', 'painful', 'weird-shape']
            },
            'responsiveness': {
                'positive': ['responsive', 'precise', 'accurate', 'snappy', 'instant'],
                'negative': ['lag', 'delay', 'unresponsive', 'sluggish', 'slow']
            },
            'connectivity': {
                'positive': ['stable-connection', 'easy-pairing', 'reliable-bluetooth'],
                'negative': ['disconnect', 'connection-issues', 'pairing-problems', 'drops-signal']
            },
            'battery': {
                'positive': ['long-battery', 'good-battery', 'lasting-power'],
                'negative': ['battery-dies', 'short-battery', 'charging-issues', 'power-hungry']
            },
            'buttons': {
                'positive': ['clicky', 'tactile', 'responsive-buttons'],
                'negative': ['mushy', 'sticky', 'unresponsive-buttons', 'broken-buttons']
            },
            'joysticks': {
                'positive': ['smooth-sticks', 'precise-analog', 'good-deadzone'],
                'negative': ['stick-drift', 'drifting', 'calibration-issues', 'dead-zone-problems']
            },
            'software': {
                'positive': ['good-software', 'customizable', 'user-friendly'],
                'negative': ['bad-software', 'buggy-app', 'confusing-interface']
            },
            'value': {
                'positive': ['worth-it', 'good-value', 'reasonable-price'],
                'negative': ['overpriced', 'not-worth', 'expensive', 'poor-value']
            }
        }
        
        # Critical issues with severity levels
        self.critical_issues = {
            'deal_breakers': ['broken', 'defective', 'useless', 'waste', 'return', 'refund'],
            'quality_concerns': ['drift', 'stopped-working', 'malfunctioning', 'died', 'failed'],
            'user_experience': ['frustrating', 'annoying', 'disappointing', 'terrible', 'awful'],
            'competitive_advantage': ['better-than', 'compared-to', 'vs', 'versus', 'alternative']
        }

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for analysis."""
        # Remove HTML tags if any
        text = re.sub(r'<[^>]+>', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\']', '', text)
        return text

    def analyze_sentiment_with_confidence(self, reviews: List[Dict]) -> Tuple[List[Dict], Counter, Counter]:
        """Analyze sentiment with improved confidence scoring and feature detection."""
        sentiments = []
        positive_themes = Counter()
        negative_themes = Counter()
        feature_scores = defaultdict(lambda: {'positive': 0, 'negative': 0})
        
        for review in reviews:
            text = self.preprocess_text(review['review_text'].lower())
            rating = review.get('review_rating', 3)
            
            # ML sentiment analysis with error handling
            ml_sentiment = None
            confidence = 0
            try:
                result = self.sentiment_analyzer(text[:500])[0]
                ml_sentiment = result['label']
                confidence = result['score']
            except Exception as e:
                logger.warning(f"Sentiment analysis failed for review: {e}")
            
            # Enhanced feature analysis
            for feature, keywords in self.feature_keywords.items():
                for sentiment_type, word_list in keywords.items():
                    for word in word_list:
                        if word.replace('-', ' ') in text or word.replace('-', '') in text:
                            if sentiment_type == 'positive':
                                positive_themes[f"{feature}_{word}"] += 1
                                feature_scores[feature]['positive'] += 1
                            else:
                                negative_themes[f"{feature}_{word}"] += 1
                                feature_scores[feature]['negative'] += 1
            
            # Critical issues detection
            for category, keywords in self.critical_issues.items():
                for word in keywords:
                    if word.replace('-', ' ') in text:
                        negative_themes[f"critical_{category}_{word}"] += 1
            
            # Determine final sentiment
            if ml_sentiment:
                if 'pos' in ml_sentiment.lower():
                    final_sentiment = 'POSITIVE'
                elif 'neg' in ml_sentiment.lower():
                    final_sentiment = 'NEGATIVE'
                else:
                    final_sentiment = 'NEUTRAL'
            else:
                # Fallback to rating-based sentiment
                if rating >= 4:
                    final_sentiment = 'POSITIVE'
                elif rating <= 2:
                    final_sentiment = 'NEGATIVE'
                else:
                    final_sentiment = 'NEUTRAL'
            
            sentiments.append({
                'rating': rating,
                'sentiment': final_sentiment,
                'confidence': confidence,
                'text_length': len(text),
                'ml_sentiment': ml_sentiment
            })
        
        return sentiments, positive_themes, negative_themes

    def chunk_and_summarize(self, text: str, max_tokens: int = 600, overlap: int = 1) -> List[str]:
        """Improved text chunking and summarization with better prompting."""
        if not text.strip():
            return ["No review content available for summarization."]
        
        try:
            sentences = sent_tokenize(text)
        except Exception as e:
            logger.error(f"Sentence tokenization failed: {e}")
            return ["Error processing text for summarization."]
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            try:
                sentence_tokens = len(self.tokenizer.encode(sentence, add_special_tokens=False))
                
                if current_tokens + sentence_tokens <= max_tokens:
                    current_chunk.append(sentence)
                    current_tokens += sentence_tokens
                else:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                    
                    # Start new chunk with overlap
                    current_chunk = current_chunk[-overlap:] + [sentence]
                    current_tokens = sum(len(self.tokenizer.encode(s, add_special_tokens=False)) 
                                       for s in current_chunk)
            except Exception as e:
                logger.warning(f"Token counting failed for sentence: {e}")
                continue
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # Generate summaries with improved prompting
        summaries = []
        for i, chunk in enumerate(chunks):
            try:
                # More specific prompt for product reviews
                prompt = f"""Analyze the following product review feedback and extract key insights about user experience, product strengths, and areas for improvement:

{chunk}

Focus on actionable insights for product development."""
                
                result = self.summarizer(
                    prompt,
                    max_length=150,
                    min_length=50,
                    do_sample=False,
                    num_beams=4
                )
                summaries.append(result[0]['summary_text'])
            except Exception as e:
                logger.error(f"Summarization failed for chunk {i}: {e}")
                summaries.append(f"Summary unavailable for chunk {i+1}")
        
        return summaries

    def generate_comprehensive_summary(self, sentiments: List[Dict], pos_themes: Counter, 
                                     neg_themes: Counter, llm_insights: List[str]) -> str:
        """Generate an enhanced comprehensive summary with actionable insights."""
        total_reviews = len(sentiments)
        if total_reviews == 0:
            return "No reviews available for analysis."
        
        # Calculate sentiment distribution
        positive_count = sum(1 for s in sentiments if s['sentiment'] == 'POSITIVE')
        negative_count = sum(1 for s in sentiments if s['sentiment'] == 'NEGATIVE')
        neutral_count = total_reviews - positive_count - negative_count
        
        # Get top themes
        top_strengths = pos_themes.most_common(5)
        top_weaknesses = neg_themes.most_common(5)
        
        # Calculate average rating
        ratings = [s['rating'] for s in sentiments if s['rating'] is not None]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        # Generate summary
        lines = []
        lines.append("=" * 70)
        lines.append("📊 COMPREHENSIVE PRODUCT REVIEW ANALYSIS")
        lines.append("=" * 70)
        lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Total Reviews Analyzed: {total_reviews}")
        lines.append(f"Average Rating: {avg_rating:.1f}/5.0")
        lines.append("")
        
        # Sentiment Overview
        lines.append("📈 SENTIMENT DISTRIBUTION")
        lines.append("-" * 30)
        lines.append(f"Positive: {positive_count} ({positive_count/total_reviews*100:.1f}%)")
        lines.append(f"Negative: {negative_count} ({negative_count/total_reviews*100:.1f}%)")
        lines.append(f"Neutral:  {neutral_count} ({neutral_count/total_reviews*100:.1f}%)")
        lines.append("")
        
        # Top Strengths
        lines.append("✅ TOP PRODUCT STRENGTHS")
        lines.append("-" * 30)
        if top_strengths:
            for theme, count in top_strengths:
                percentage = (count / total_reviews) * 100
                lines.append(f"• {theme.replace('_', ' ').title()}: {count} mentions ({percentage:.1f}%)")
        else:
            lines.append("• No significant strengths identified")
        lines.append("")
        
        # Top Weaknesses
        lines.append("❌ TOP AREAS FOR IMPROVEMENT")
        lines.append("-" * 30)
        if top_weaknesses:
            for theme, count in top_weaknesses:
                percentage = (count / total_reviews) * 100
                lines.append(f"• {theme.replace('_', ' ').title()}: {count} mentions ({percentage:.1f}%)")
        else:
            lines.append("• No significant weaknesses identified")
        lines.append("")
        
        # AI-Generated Insights
        lines.append("🤖 AI-GENERATED INSIGHTS")
        lines.append("-" * 30)
        for i, insight in enumerate(llm_insights, 1):
            lines.append(f"{i}. {insight}")
        lines.append("")
        
        # Recommendations
        lines.append("💡 RECOMMENDATIONS")
        lines.append("-" * 30)
        
        # Priority recommendations based on negative sentiment
        critical_issues = [theme for theme, _ in top_weaknesses if 'critical' in theme]
        if critical_issues:
            lines.append("🔴 CRITICAL ISSUES (Immediate attention required):")
            for issue in critical_issues[:3]:
                lines.append(f"   • Address {issue.replace('critical_', '').replace('_', ' ')}")
        
        # Feature improvement recommendations
        if negative_count > positive_count:
            lines.append("🟡 PRIORITY IMPROVEMENTS:")
            lines.append("   • Focus on addressing top negative themes")
            lines.append("   • Conduct user testing for identified pain points")
            lines.append("   • Consider product redesign for critical issues")
        else:
            lines.append("🟢 ENHANCEMENT OPPORTUNITIES:")
            lines.append("   • Leverage positive feedback in marketing")
            lines.append("   • Continue improving identified strength areas")
        
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)

    def export_results(self, results: Dict, filename: str = "review_analysis.json"):
        """Export analysis results to JSON file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results exported to {filename}")
        except Exception as e:
            logger.error(f"Failed to export results: {e}")

def parse_reviews_from_file(file_path: str) -> Optional[List[Dict]]:
    """Enhanced HTML parsing with better error handling."""
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
        
        # Try multiple selectors for different review formats
        review_selectors = [
            'li.review-item',
            '.review-item',
            '.review',
            '[data-testid="review"]'
        ]
        
        review_elements = []
        for selector in review_selectors:
            elements = soup.select(selector)
            if elements:
                review_elements = elements
                break
        
        if not review_elements:
            logger.warning("No review elements found with standard selectors")
            return []
        
        for element in review_elements:
            review_data = {'review_text': '', 'review_rating': None}
            
            # Extract review text
            text_selectors = [
                'p.pre-white-space',
                '.review-text',
                'p',
                '.content'
            ]
            
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
            
            # Try alternative rating extraction
            if not review_data['review_rating']:
                star_elements = element.select('[class*="star"], [data-rating]')
                for star_elem in star_elements:
                    rating_text = star_elem.get('data-rating') or star_elem.get_text()
                    rating_match = re.search(r'(\d(?:\.\d)?)', str(rating_text))
                    if rating_match:
                        review_data['review_rating'] = float(rating_match.group(1))
                        break
            
            if review_data['review_text']:  # Only add if we have text
                reviews.append(review_data)
        
        logger.info(f"Successfully parsed {len(reviews)} reviews")
        return reviews
    
    except Exception as e:
        logger.error(f"Error parsing HTML: {e}")
        return None

def main():
    """Main execution function with enhanced error handling."""
    try:
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        
        # Parse reviews from file
        reviews = parse_reviews_from_file('sample.html')
        if not reviews:
            print("❌ No reviews found or error parsing file.")
            return
        
        print(f"📚 Found {len(reviews)} reviews to analyze...")
        
        # Initialize analyzer
        analyzer = ProductReviewAnalyzer()
        
        # Perform analysis
        print("🔍 Analyzing sentiments and themes...")
        sentiments, pos_themes, neg_themes = analyzer.analyze_sentiment_with_confidence(reviews)
        
        print("📝 Generating summaries...")
        combined_text = ' '.join(r['review_text'] for r in reviews if r['review_text'])
        llm_insights = analyzer.chunk_and_summarize(combined_text)
        
        print("📊 Generating comprehensive summary...")
        summary = analyzer.generate_comprehensive_summary(sentiments, pos_themes, neg_themes, llm_insights)
        
        # Display results
        print("\n" + summary)
        
        # Export results
        results = {
            'total_reviews': len(reviews),
            'sentiments': sentiments,
            'positive_themes': dict(pos_themes.most_common(10)),
            'negative_themes': dict(neg_themes.most_common(10)),
            'llm_insights': llm_insights,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        analyzer.export_results(results)
        print("\n✅ Analysis complete! Results exported to review_analysis.json")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"❌ Analysis failed: {e}")

if __name__ == "__main__":
    main()