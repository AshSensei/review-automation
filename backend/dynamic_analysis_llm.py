import json
import re
from collections import Counter, defaultdict
from bs4 import BeautifulSoup
from transformers import pipeline
import torch
import logging
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy
from textstat import flesch_reading_ease
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedReviewAnalyzer:
    def __init__(self):
        """Initialize the analyzer with better models and approaches."""
        self.device = 0 if torch.cuda.is_available() else -1
        
        # Initialize models
        try:
            # Use a proper text classification model
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=self.device
            )
            
            # Use a better model for text analysis
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=self.device
            )
            
            # Text summarization for key insights
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=self.device,
                max_length=150,
                min_length=30,
                do_sample=False
            )
            
            logger.info(f"Models loaded successfully on device: {self.device}")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            # Fallback to simpler approach
            self.sentiment_analyzer = pipeline("sentiment-analysis", device=self.device)
            self.classifier = None
            self.summarizer = None
    
    def analyze_themes_with_classification(self, reviews: List[Dict], product_type: str = "gaming controller") -> Dict[str, Any]:
        """Use zero-shot classification to identify themes dynamically."""
        
        # Define potential themes based on product type
        theme_candidates = {
            "gaming controller": [
                "build quality", "button responsiveness", "connectivity", "battery life",
                "ergonomics", "durability", "price value", "compatibility", 
                "wireless performance", "haptic feedback", "design", "software"
            ],
            "headphones": [
                "sound quality", "comfort", "build quality", "noise cancellation",
                "battery life", "connectivity", "price value", "durability",
                "microphone quality", "design", "portability"
            ],
            "laptop": [
                "performance", "battery life", "build quality", "display quality",
                "keyboard", "price value", "portability", "heating", "design",
                "connectivity", "software", "customer service"
            ],
            "default": [
                "quality", "performance", "design", "price", "durability",
                "ease of use", "customer service", "value", "reliability"
            ]
        }
        
        themes = theme_candidates.get(product_type.lower(), theme_candidates["default"])
        
        review_texts = [r['review_text'] for r in reviews if r['review_text'].strip()]
        
        if not review_texts:
            return {"themes": {}, "theme_sentiments": {}}
        
        # Analyze themes using classification
        theme_analysis = {}
        theme_sentiments = {}
        
        if self.classifier:
            try:
                # Sample reviews for analysis
                sample_reviews = review_texts[:15]  # Limit to avoid timeouts
                
                for i, review in enumerate(sample_reviews):
                    # Truncate review to manageable length
                    truncated_review = review[:500]
                    
                    # Classify themes
                    result = self.classifier(truncated_review, themes)
                    
                    # Get sentiment for this review
                    sentiment = self.sentiment_analyzer(truncated_review[:512])[0]
                    
                    # Store top themes for this review
                    for label, score in zip(result['labels'][:3], result['scores'][:3]):
                        if score > 0.3:  # Threshold for relevance
                            if label not in theme_analysis:
                                theme_analysis[label] = []
                                theme_sentiments[label] = []
                            
                            theme_analysis[label].append({
                                'review_index': i,
                                'score': score,
                                'text_snippet': truncated_review[:200] + "..."
                            })
                            
                            theme_sentiments[label].append({
                                'sentiment': sentiment['label'],
                                'confidence': sentiment['score'],
                                'score': score
                            })
                
                logger.info(f"Analyzed themes for {len(sample_reviews)} reviews")
                
            except Exception as e:
                logger.error(f"Theme classification failed: {e}")
                # Fallback to keyword-based analysis
                return self.fallback_theme_analysis(reviews, themes)
        
        return {
            "themes": theme_analysis,
            "theme_sentiments": theme_sentiments
        }
    
    def fallback_theme_analysis(self, reviews: List[Dict], themes: List[str]) -> Dict[str, Any]:
        """Fallback keyword-based theme analysis."""
        
        # Create keyword mappings for themes
        theme_keywords = {
            "build quality": ["build", "quality", "construction", "material", "solid", "cheap", "flimsy"],
            "battery life": ["battery", "charge", "charging", "power", "died", "drain"],
            "connectivity": ["connect", "bluetooth", "wireless", "lag", "disconnect", "pair"],
            "comfort": ["comfort", "comfortable", "ergonomic", "fit", "wear", "hurt"],
            "performance": ["performance", "speed", "fast", "slow", "responsive", "lag"],
            "design": ["design", "look", "appearance", "aesthetic", "color", "style"],
            "price value": ["price", "cost", "value", "worth", "expensive", "cheap", "money"],
            "durability": ["durable", "break", "broke", "broken", "last", "lasting", "wear"]
        }
        
        review_texts = [r['review_text'].lower() for r in reviews if r['review_text'].strip()]
        
        theme_analysis = {}
        theme_sentiments = {}
        
        for theme in themes:
            theme_mentions = []
            theme_sentiment_scores = []
            
            keywords = theme_keywords.get(theme, [theme.lower()])
            
            for i, review in enumerate(review_texts):
                # Count keyword matches
                matches = sum(1 for keyword in keywords if keyword in review)
                
                if matches > 0:
                    # Get sentiment for this review
                    sentiment = self.sentiment_analyzer(review[:512])[0]
                    
                    theme_mentions.append({
                        'review_index': i,
                        'matches': matches,
                        'text_snippet': review[:200] + "..."
                    })
                    
                    theme_sentiment_scores.append({
                        'sentiment': sentiment['label'],
                        'confidence': sentiment['score'],
                        'matches': matches
                    })
            
            if theme_mentions:
                theme_analysis[theme] = theme_mentions
                theme_sentiments[theme] = theme_sentiment_scores
        
        return {
            "themes": theme_analysis,
            "theme_sentiments": theme_sentiments
        }
    
    def extract_specific_issues(self, reviews: List[Dict]) -> List[Dict[str, Any]]:
        """Extract specific issues using pattern matching and sentiment analysis."""
        
        issues = []
        
        # Common issue patterns
        issue_patterns = [
            (r"(break|broke|broken|fail|failed|stop|stopped)\s+(\w+)", "failure"),
            (r"(doesn't|don't|won't|can't)\s+(\w+)", "functionality"),
            (r"(poor|bad|terrible|awful|horrible)\s+(\w+)", "quality"),
            (r"(cheap|flimsy|weak|fragile)", "build_quality"),
            (r"(expensive|overpriced|costly)", "price"),
            (r"(uncomfortable|hurt|pain)", "comfort"),
            (r"(lag|delay|slow|unresponsive)", "performance"),
            (r"(disconnect|connection|connectivity)", "connectivity")
        ]
        
        # Find negative reviews
        negative_reviews = []
        for review in reviews:
            if review.get('review_rating', 3) <= 2:
                negative_reviews.append(review)
            else:
                # Check sentiment
                sentiment = self.sentiment_analyzer(review['review_text'][:512])[0]
                if 'neg' in sentiment['label'].lower() and sentiment['score'] > 0.7:
                    negative_reviews.append(review)
        
        # Extract issues from negative reviews
        for review in negative_reviews:
            text = review['review_text'].lower()
            
            for pattern, issue_type in issue_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    if isinstance(match, tuple):
                        issue_description = f"{match[0]} {match[1]}"
                    else:
                        issue_description = match
                    
                    # Get context around the issue
                    context_start = max(0, text.find(issue_description) - 50)
                    context_end = min(len(text), text.find(issue_description) + 100)
                    context = text[context_start:context_end]
                    
                    issues.append({
                        'type': issue_type,
                        'description': issue_description,
                        'context': context,
                        'review_rating': review.get('review_rating', 'unknown'),
                        'severity': 'high' if review.get('review_rating', 3) <= 2 else 'medium'
                    })
        
        # Remove duplicates and rank by frequency
        issue_counts = Counter(issue['description'] for issue in issues)
        unique_issues = []
        
        for issue in issues:
            if issue['description'] not in [ui['description'] for ui in unique_issues]:
                issue['frequency'] = issue_counts[issue['description']]
                unique_issues.append(issue)
        
        # Sort by frequency and severity
        unique_issues.sort(key=lambda x: (x['frequency'], x['severity'] == 'high'), reverse=True)
        
        return unique_issues[:10]  # Return top 10 issues
    
    def generate_insights_summary(self, reviews: List[Dict]) -> str:
        """Generate a summary of key insights."""
        
        if not self.summarizer:
            return "Summary generation not available"
        
        # Combine all reviews for summarization
        all_text = " ".join([r['review_text'] for r in reviews if r['review_text'].strip()])
        
        # Truncate to manageable length
        if len(all_text) > 5000:
            all_text = all_text[:5000]
        
        try:
            summary = self.summarizer(all_text, max_length=130, min_length=50, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return "Could not generate summary"
    
    def calculate_review_metrics(self, reviews: List[Dict]) -> Dict[str, Any]:
        """Calculate various metrics about the reviews."""
        
        ratings = [r.get('review_rating', 3) for r in reviews]
        review_lengths = [len(r['review_text']) for r in reviews if r['review_text'].strip()]
        
        # Sentiment analysis
        sentiments = []
        for review in reviews:
            if review['review_text'].strip():
                sentiment = self.sentiment_analyzer(review['review_text'][:512])[0]
                sentiments.append(sentiment)
        
        sentiment_counts = Counter(s['label'] for s in sentiments)
        
        # Reading complexity
        avg_reading_ease = 0
        if review_lengths:
            try:
                combined_text = " ".join([r['review_text'] for r in reviews[:10]])
                avg_reading_ease = flesch_reading_ease(combined_text)
            except:
                avg_reading_ease = 0
        
        return {
            'total_reviews': len(reviews),
            'average_rating': sum(ratings) / len(ratings) if ratings else 0,
            'rating_distribution': dict(Counter(ratings)),
            'average_review_length': sum(review_lengths) / len(review_lengths) if review_lengths else 0,
            'sentiment_distribution': dict(sentiment_counts),
            'reading_ease_score': avg_reading_ease
        }
    
    def generate_comprehensive_analysis(self, reviews: List[Dict], product_type: str = "product") -> Dict[str, Any]:
        """Generate comprehensive analysis with actionable insights."""
        
        print("🔍 Analyzing themes...")
        theme_analysis = self.analyze_themes_with_classification(reviews, product_type)
        
        print("🔍 Extracting issues...")
        issues = self.extract_specific_issues(reviews)
        
        print("🔍 Calculating metrics...")
        metrics = self.calculate_review_metrics(reviews)
        
        print("🔍 Generating summary...")
        summary = self.generate_insights_summary(reviews)
        
        return {
            'themes': theme_analysis,
            'issues': issues,
            'metrics': metrics,
            'summary': summary,
            'analysis_metadata': {
                'total_reviews': len(reviews),
                'analysis_date': datetime.now().isoformat(),
                'product_type': product_type
            }
        }
    
    def create_detailed_report(self, analysis_results: Dict[str, Any]) -> str:
        """Create a detailed, actionable report."""
        
        themes = analysis_results['themes']
        issues = analysis_results['issues']
        metrics = analysis_results['metrics']
        summary = analysis_results['summary']
        metadata = analysis_results['analysis_metadata']
        
        lines = []
        lines.append("=" * 80)
        lines.append("📊 IMPROVED PRODUCT REVIEW ANALYSIS")
        lines.append("=" * 80)
        lines.append(f"Product: {metadata['product_type'].title()}")
        lines.append(f"Reviews Analyzed: {metadata['total_reviews']}")
        lines.append(f"Analysis Date: {metadata['analysis_date']}")
        lines.append("")
        
        # Executive Summary
        lines.append("📋 EXECUTIVE SUMMARY")
        lines.append("-" * 40)
        lines.append(summary)
        lines.append("")
        
        # Key Metrics
        lines.append("📈 KEY METRICS")
        lines.append("-" * 40)
        lines.append(f"Average Rating: {metrics['average_rating']:.1f}/5")
        lines.append(f"Total Reviews: {metrics['total_reviews']}")
        lines.append(f"Average Review Length: {metrics['average_review_length']:.0f} characters")
        lines.append("")
        
        # Sentiment Distribution
        lines.append("Sentiment Distribution:")
        total_sentiment = sum(metrics['sentiment_distribution'].values())
        for sentiment, count in metrics['sentiment_distribution'].items():
            percentage = (count / total_sentiment * 100) if total_sentiment > 0 else 0
            emoji = "😊" if "pos" in sentiment.lower() else "😞" if "neg" in sentiment.lower() else "😐"
            lines.append(f"  {emoji} {sentiment}: {count} ({percentage:.1f}%)")
        lines.append("")
        
        # Rating Distribution
        lines.append("Rating Distribution:")
        for rating, count in sorted(metrics['rating_distribution'].items()):
            stars = "⭐" * int(rating) if rating > 0 else "No rating"
            lines.append(f"  {stars} {rating}: {count} reviews")
        lines.append("")
        
        # Top Issues
        lines.append("🔴 TOP ISSUES IDENTIFIED")
        lines.append("-" * 40)
        if issues:
            for i, issue in enumerate(issues[:5], 1):
                lines.append(f"{i}. {issue['description'].title()}")
                lines.append(f"   Type: {issue['type'].replace('_', ' ').title()}")
                lines.append(f"   Frequency: {issue['frequency']} mentions")
                lines.append(f"   Severity: {issue['severity'].title()}")
                lines.append(f"   Context: {issue['context'][:100]}...")
                lines.append("")
        else:
            lines.append("No significant issues identified")
            lines.append("")
        
        # Theme Analysis
        lines.append("🎯 THEME ANALYSIS")
        lines.append("-" * 40)
        
        if themes.get('themes'):
            for theme, mentions in themes['themes'].items():
                if mentions:  # Only show themes with mentions
                    lines.append(f"📌 {theme.title()}")
                    lines.append(f"   Mentions: {len(mentions)}")
                    
                    # Show sentiment for this theme
                    if theme in themes.get('theme_sentiments', {}):
                        theme_sentiments = themes['theme_sentiments'][theme]
                        positive = sum(1 for s in theme_sentiments if 'pos' in s['sentiment'].lower())
                        negative = sum(1 for s in theme_sentiments if 'neg' in s['sentiment'].lower())
                        neutral = len(theme_sentiments) - positive - negative
                        
                        lines.append(f"   Sentiment: {positive} positive, {negative} negative, {neutral} neutral")
                    
                    # Show example
                    if mentions:
                        example = mentions[0]['text_snippet'][:150] + "..."
                        lines.append(f"   Example: {example}")
                    lines.append("")
        
        # Recommendations
        lines.append("💡 RECOMMENDATIONS")
        lines.append("-" * 40)
        
        # Generate recommendations based on issues and themes
        recommendations = []
        
        # Based on issues
        if issues:
            top_issue = issues[0]
            if top_issue['type'] == 'build_quality':
                recommendations.append("Consider improving build quality and materials")
            elif top_issue['type'] == 'connectivity':
                recommendations.append("Focus on improving wireless connectivity and pairing")
            elif top_issue['type'] == 'performance':
                recommendations.append("Optimize performance and reduce lag/delays")
            elif top_issue['type'] == 'comfort':
                recommendations.append("Redesign ergonomics for better comfort")
            elif top_issue['type'] == 'price':
                recommendations.append("Review pricing strategy or improve value proposition")
        
        # Based on sentiment
        if metrics['sentiment_distribution'].get('NEGATIVE', 0) > metrics['sentiment_distribution'].get('POSITIVE', 0):
            recommendations.append("Address negative feedback patterns urgently")
        
        # Based on ratings
        if metrics['average_rating'] < 3.5:
            recommendations.append("Overall satisfaction is below average - comprehensive review needed")
        elif metrics['average_rating'] > 4.0:
            recommendations.append("Good overall satisfaction - focus on maintaining quality")
        
        if not recommendations:
            recommendations.append("Continue monitoring customer feedback for emerging issues")
        
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. {rec}")
        
        lines.append("")
        lines.append("=" * 80)
        lines.append("✅ Analysis Complete")
        lines.append("=" * 80)
        
        return "\n".join(lines)

# Keep the same HTML parsing function
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
        
        # Initialize improved analyzer
        print("🤖 Initializing improved analyzer...")
        analyzer = ImprovedReviewAnalyzer()
        
        # Generate comprehensive analysis
        print("🔍 Running comprehensive analysis...")
        analysis_results = analyzer.generate_comprehensive_analysis(reviews, product_type)
        
        # Generate and display report
        print("📊 Generating detailed report...")
        report = analyzer.create_detailed_report(analysis_results)
        
        print("\n" + report)
        
        # Export results
        with open('improved_analysis_results.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False, default=str)
        
        print("\n✅ Analysis complete! Results exported to improved_analysis_results.json")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"❌ Analysis failed: {e}")

if __name__ == "__main__":
    main()