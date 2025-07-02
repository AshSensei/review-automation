import json
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer
import nltk
import torch
from nltk.tokenize import sent_tokenize
print(torch.__version__)
# Device selection
device = 0 if torch.cuda.is_available() else -1
print(f"Device set to use {'GPU' if device == 0 else 'CPU'}")
import sys
print("🔍 Python path:", sys.executable)

# Download NLTK sentence tokenizer if needed
# nltk.download('punkt')

def chunk_by_tokens_with_overlap(text, tokenizer, max_tokens=900, overlap_sentences=2):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_tokens = 0

    for i, sentence in enumerate(sentences):
        sentence_tokens = len(tokenizer.encode(sentence, add_special_tokens=False))
        if current_tokens + sentence_tokens <= max_tokens:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        else:
            chunks.append(" ".join(current_chunk))
            overlap_slice = current_chunk[-overlap_sentences:] if overlap_sentences < len(current_chunk) else current_chunk
            current_chunk = overlap_slice + [sentence]
            current_tokens = sum(len(tokenizer.encode(s, add_special_tokens=False)) for s in current_chunk)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def parse_reviews_from_file(file_path, min_review_length=10):
    with open(file_path, "r", encoding="utf-8") as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, "html.parser")

    script_tag = soup.find("script", type="application/ld+json")
    json_ld = json.loads(script_tag.string) if script_tag else {}
    total_rating = json_ld.get("aggregateRating", {}).get("ratingValue")

    review_elements = soup.find_all("li", class_="review-item")
    reviews = []

    for review in review_elements:
        body_tag = review.find("p", class_="pre-white-space")
        review_text = body_tag.get_text(strip=True) if body_tag else ""

        if len(review_text.split()) < min_review_length:
            continue

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

def summarize_chunks(chunks, summarizer):
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"🧠 Summarizing chunk {i+1}/{len(chunks)}...")
        structured_prompt = (
            "Summarize the following customer reviews of a game controller.\n"
            "List the common opinions clearly:\n"
            "- Positives (features, usability, build quality, etc.)\n"
            "- Criticisms (problems, missing features, complaints)\n\n"
            f"{chunk}"
        )
        summary = summarizer(
            structured_prompt,
            max_length=300,
            min_length=100,
            do_sample=False,
            early_stopping=True,
        )
        chunk_summaries.append(summary[0]['summary_text'])
    return chunk_summaries

def summarize_final_summary(summary_chunks, summarizer):
    final_summary_parts = []
    for i, chunk in enumerate(summary_chunks):
        print(f"🔁 Refining summary chunk {i+1}/{len(summary_chunks)}...")
        final_prompt = (
            "You are summarizing feedback on a game controller based on aggregated summaries from multiple users.\n"
            "Present the output in this format:\n"
            "- Key Strengths\n- Key Weaknesses\n- Overall Verdict\n\n"
            f"{chunk}"
        )
        summary = summarizer(
            final_prompt,
            max_length=300,
            min_length=120,
            do_sample=False,
            early_stopping=True,
        )
        final_summary_parts.append(summary[0]['summary_text'])
    return "\n".join(final_summary_parts)

if __name__ == "__main__":
    input_file = "sample.html"
    output_file = "review_data.json"

    data = parse_reviews_from_file(input_file)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Extracted {len(data['reviews'])} reviews.")
    print(f"📄 Total Rating: {data['total_rating']}")
    print(f"💾 Review data saved to {output_file}")

    combined_text = " ".join([review["review_text"] for review in data["reviews"]])

    # Model setup - switch to stronger model
    model_name = "philschmid/bart-large-cnn-samsum"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    summarizer = pipeline(
        "summarization",
        model=model_name,
        tokenizer=tokenizer,
        device=device,
        framework="pt"
    )

    # Chunk and summarize
    chunks = chunk_by_tokens_with_overlap(combined_text, tokenizer, max_tokens=900, overlap_sentences=2)
    print(f"📦 Total chunks to summarize: {len(chunks)}")

    chunk_summaries = summarize_chunks(chunks, summarizer)

    # Final summarization step
    final_input = "\n\n".join(f"Summary {i+1}:\n{s}" for i, s in enumerate(chunk_summaries))
    summary_chunks = chunk_by_tokens_with_overlap(final_input, tokenizer, max_tokens=900, overlap_sentences=1)
    print(f"\n🧪 Final summarization step - {len(summary_chunks)} chunks")

    final_summary_text = summarize_final_summary(summary_chunks, summarizer)

    # Output final result
    print("\n📌 Final Summary:")
    print(final_summary_text)
