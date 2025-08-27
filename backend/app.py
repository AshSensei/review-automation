import os
import gc
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

# Import from BOTH analyzer files
from openai_llm_analysis import (
    OpenAIReviewAnalyzer,
    parse_reviews_comprehensive,
    ANALYSIS_REVIEW_LIMIT,
)

from competitor_analysis import (
    OpenAIHybridReviewAnalyzer,
    parse_reviews_comprehensive as parse_reviews_hybrid,  # Alias to avoid conflicts
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask App and CORS
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})


# === SINGLE PRODUCT ANALYSIS (Non-Hybrid) ===
@app.route("/api/analyze-html", methods=["POST"])
def analyze_html_endpoint():
    """Analyzes the HTML of a single product page for reviews using the standard analyzer."""
    try:
        # Get HTML from request, matching the target file
        data = request.get_json(force=True)
        html_content = data.get("html", "")
        product_type = data.get("product_type", "product")

        if not html_content:
            return jsonify({"error": "No HTML content provided"}), 400

        # Use the comprehensive parsing function from openai_llm_analysis
        reviews = parse_reviews_comprehensive(html_content)

        if not reviews:
            return jsonify({"error": "No reviews found in HTML"}), 400

        # Limit reviews for analysis
        reviews = reviews[:ANALYSIS_REVIEW_LIMIT]

        # Initialize analyzer
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            # Match the error message from the target file
            return jsonify({"error": "OpenAI API key not configured"}), 500

        analyzer = OpenAIReviewAnalyzer(openai_api_key)

        # Generate comprehensive analysis
        analysis_results = analyzer.generate_comprehensive_analysis(
            reviews, product_type
        )

        # The response structure is already a perfect match
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

        gc.collect()
        return jsonify(response)

    except Exception as e:
        # Match the more detailed logging and error response from the target file
        logger.error(f"API analysis failed: {e}", exc_info=True)
        return (
            jsonify({"error": "Analysis failed", "message": str(e), "success": False}),
            500,
        )


# === COMPETITIVE ANALYSIS (Hybrid) ===
@app.route("/api/compare", methods=["POST"])
def compare_competitive_endpoint():
    """Compares two products using the hybrid analyzer for competitive analysis."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        html_a = data.get("html_a", "")
        html_b = data.get("html_b", "")
        product_name_a = data.get("product_name_a", "Product A")
        product_name_b = data.get("product_name_b", "Product B")

        if not html_a or not html_b:
            return jsonify({"error": "Both html_a and html_b must be provided"}), 400

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            return (
                jsonify({"error": "OpenAI API key not configured on the server"}),
                500,
            )

        # Parse reviews for both products
        reviews_a = parse_reviews_hybrid(html_a)
        reviews_b = parse_reviews_hybrid(html_b)

        if not reviews_a:
            return (
                jsonify(
                    {"error": f"Could not extract reviews from {product_name_a} HTML"}
                ),
                400,
            )
        if not reviews_b:
            return (
                jsonify(
                    {"error": f"Could not extract reviews from {product_name_b} HTML"}
                ),
                400,
            )

        # Limit reviews for analysis
        reviews_a = reviews_a[:ANALYSIS_REVIEW_LIMIT]
        reviews_b = reviews_b[:ANALYSIS_REVIEW_LIMIT]

        # Initialize the analyzer from the other file
        analyzer = OpenAIHybridReviewAnalyzer(openai_api_key)

        # Analyze both products
        logger.info(f"Analyzing {product_name_a}...")
        analysis_a = analyzer.generate_comprehensive_analysis(reviews_a, product_name_a)

        logger.info(f"Analyzing {product_name_b}...")
        analysis_b = analyzer.generate_comprehensive_analysis(reviews_b, product_name_b)

        # Use the analyzer's comparison method
        logger.info("Generating competitive comparison insights...")
        comparison_results = analyzer.compare_products(analysis_a, analysis_b)

        # --- 2. CREATE THE CORRECT FINAL RESPONSE STRUCTURE ---
        # This structure matches what your React frontend expects.
        final_response = {
            "product_a": analysis_a,
            "product_b": analysis_b,
            "comparison": comparison_results,
        }

        gc.collect()
        return jsonify(final_response)

    except Exception as e:
        logger.error(f"Error in /api/compare: {e}", exc_info=True)
        return (
            jsonify({"error": "An internal server error occurred", "message": str(e)}),
            500,
        )


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


# Main execution
if __name__ == "__main__":
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY environment variable not set!")

    # Use port 5000 for main app
    port = int(os.getenv("PORT", 5000))
    # host='0.0.0.0' is crucial for Render deployment
    app.run(debug=True, port=port, host="0.0.0.0")
