"""
Book Passage Analyzer

A script which analyzes book passages — it counts words, detects the
dominant emotion, tries to figure out which book a passage might be from,
and generates a short summary.

Groq's free API through LiteLLM is used here. The idea was to keep it
simple: one file, minimal dependencies, and it just works.

Usage:
    python analyzer.py                        # interactive mode
    python analyzer.py "some passage here"    # pass text directly
    python analyzer.py --file passage.txt     # read from a file

Author: Satish Prem Anand
Date: May 2026
"""

import os
import sys
import json
from dotenv import load_dotenv
from litellm import completion

load_dotenv()


# Models I'm using, in order of preference. If the first one hits a rate
# limit or is down for some reason, LiteLLM will automatically try the next.
# Hasn't failed me yet in other projects as well with 4 options in the chain.

MODELS = [
    "groq/llama-3.1-8b-instant",
    "groq/meta-llama/llama-4-scout-17b-16e-instruct",
    "groq/gemma2-9b-it",
    "groq/llama-guard-3-8b",
]


def count_words(text):
    """Just splits on whitespace and counts. No need to overthink this one."""
    words = text.split()
    return len(words)


def call_llm(prompt, system_message="You are a helpful literary analyst."):
    """
    Sends a prompt to Groq through LiteLLM. Tries each model in the fallback
    chain until one responds. If they all fail, raises an exception — but
    that's never happened in practice.
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]

    for i, model in enumerate(MODELS):
        try:
            response = completion(
                model=model,
                messages=messages,
                temperature=0.3,  # keeping it deterministic
                max_tokens=1024,
            )
            return response.choices[0].message.content

        except Exception as e:
            if i < len(MODELS) - 1:
                print(f"  ⚠ {model} unavailable ({str(e)[:50]}...), trying next model...")
            else:
                raise Exception(f"All models failed. Last error: {e}")


def detect_emotion(text):
    """
    Asks the LLM to identify the dominant emotion in the passage.
    Returns a dict with the emotion, confidence level, and reasoning.
    """
    prompt = f"""Analyze the predominant emotion in this passage. 
Pick from: joy, sadness, anger, fear, surprise, disgust, love, hope, melancholy, nostalgia, or any other fitting emotion.

Respond in this exact JSON format (nothing else):
{{
    "emotion": "<the dominant emotion>",
    "confidence": "<your confidence as a percentage>",
    "reasoning": "<1-2 sentences explaining why you chose this emotion>"
}}

Passage:
\"{text}\""""

    response = call_llm(prompt)

    try:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"emotion": "unknown", "confidence": "N/A", "reasoning": response}


def identify_books(text):
    """
    This is where the LLM earns its keep — it pattern-matches writing style,
    themes, and vocabulary against its training data to guess possible sources.
    Returns a list of book suggestions with reasons.
    """
    prompt = f"""Based on the writing style, themes, vocabulary, and content of this passage, 
suggest 2-3 books it might be from.

Respond in this exact JSON format (nothing else):
[
    {{"title": "<book title>", "author": "<author name>", "reason": "<why this book>"}},
    {{"title": "<book title>", "author": "<author name>", "reason": "<why this book>"}}
]

Passage:
\"{text}\""""

    response = call_llm(prompt)

    try:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return [{"title": "Could not parse", "author": "N/A", "reason": response}]


def summarize_passage(text):
    """Condenses the passage into 2-3 sentences. Straightforward."""
    prompt = f"""Summarize this passage in 2-3 clear, concise sentences. 
Capture the main idea, tone, and any key events or themes.

Passage:
\"{text}\""""

    return call_llm(prompt)


def analyze_passage(text):
    """
    The main function that ties it all together. Runs each analysis step,
    prints a formatted report, and saves everything to JSON.
    """
    print("\n" + "═" * 66)
    print("                    📖 BOOK PASSAGE ANALYZER")
    print("═" * 66)

    # Word count — no API call needed for this
    word_count = count_words(text)
    print(f"\n📊 WORD COUNT")
    print(f"   Total words: {word_count}")

    # Emotion detection
    print(f"\n🎭 EMOTION ANALYSIS")
    print(f"   Analyzing...")
    emotion = detect_emotion(text)
    print(f"   Emotion   : {emotion.get('emotion', 'N/A').title()}")
    print(f"   Confidence: {emotion.get('confidence', 'N/A')}")
    print(f"   Reasoning : {emotion.get('reasoning', 'N/A')}")

    # Book identification
    print(f"\n📚 POSSIBLE BOOKS")
    print(f"   Searching...")
    books = identify_books(text)
    for i, book in enumerate(books, 1):
        print(f"   {i}. \"{book.get('title', 'Unknown')}\" by {book.get('author', 'Unknown')}")
        print(f"      → {book.get('reason', 'N/A')}")

    # Summary
    print(f"\n📝 SUMMARY")
    print(f"   Generating...")
    summary = summarize_passage(text)
    wrapped = summary.replace(". ", ".\n   ")
    print(f"   {wrapped}")

    print("\n" + "═" * 66)

    # Save to JSON so results can be used elsewhere if needed
    results = {
        "passage": text[:200] + "..." if len(text) > 200 else text,
        "word_count": word_count,
        "emotion": emotion,
        "possible_books": books,
        "summary": summary,
    }

    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("💾 Results saved to analysis_results.json")
    print("═" * 66 + "\n")

    return results


if __name__ == "__main__":

    # Quick check — make sure the API key exists before we do anything
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY not found!")
        print("   Create a .env file with: GROQ_API_KEY=your_key_here")
        print("   Get your key at: https://console.groq.com/keys")
        sys.exit(1)

    # Figure out where the input is coming from
    if len(sys.argv) > 1:
        if sys.argv[1] == "--file":
            if len(sys.argv) < 3:
                print("❌ Usage: python analyzer.py --file <path_to_file>")
                sys.exit(1)
            filepath = sys.argv[2]
            if not os.path.exists(filepath):
                print(f"❌ File not found: {filepath}")
                sys.exit(1)
            with open(filepath, "r") as f:
                passage = f.read().strip()
            print(f"📄 Reading from: {filepath}")
        else:
            passage = " ".join(sys.argv[1:])
    else:
        # Interactive mode — let the user paste text in
        print("\n📖 Book Passage Analyzer")
        print("─" * 40)
        print("Paste your passage below (press Enter twice when done):\n")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        passage = "\n".join(lines)

    if not passage.strip():
        print("❌ No text provided. Nothing to analyze!")
        sys.exit(1)

    analyze_passage(passage)
