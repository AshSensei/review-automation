import json
from bs4 import BeautifulSoup

def parse_reviews_from_file(file_path):
    # Load the HTML content
    with open(file_path, "r", encoding="utf-8") as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, "html.parser")

    # Extract total product rating from JSON-LD metadata
    script_tag = soup.find("script", type="application/ld+json")
    json_ld = json.loads(script_tag.string) if script_tag else {}
    total_rating = json_ld.get("aggregateRating", {}).get("ratingValue")

    # Extract individual reviews
    review_elements = soup.find_all("li", class_="review-item")
    reviews = []

    for review in review_elements:
        # Extract review body
        body_tag = review.find("p", class_="pre-white-space")
        review_text = body_tag.get_text(strip=True) if body_tag else ""

        # Extract rating from visually-hidden tag
        rating_tag = review.find("p", class_="visually-hidden")
        rating = None
        if rating_tag and "Rated" in rating_tag.text:
            try:
                rating = float(rating_tag.text.strip().split(" ")[1])
            except (IndexError, ValueError):
                rating = None

        reviews.append({
            "review_text": review_text,
            "review_rating": rating
        })

    return {
        "total_rating": float(total_rating) if total_rating else None,
        "reviews": reviews
    }

if __name__ == "__main__":
    input_file = "sample.html"          # Your HTML file with Best Buy reviews
    output_file = "review_data.json"    # Output file to save extracted data

    data = parse_reviews_from_file(input_file)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Extracted {len(data['reviews'])} reviews.")
    print(f"📄 Total Rating: {data['total_rating']}")
    print(f"💾 Data saved to {output_file}")
