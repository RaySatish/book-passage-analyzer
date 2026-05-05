"""
Book Passage Analyzer
=====================
Hey! This script takes any book passage you throw at it and tells you:
  1. How many words are in it
  2. What emotion it carries (joy, sadness, anger, etc.)
  3. Which 2-3 books it might be from
  4. A quick 2-3 sentence summary

How it works:
  - Word count is plain Python (no need for AI here, right?)
  - For emotion, book matching, and summary — we use Groq's LLMs
  - If one model hits a rate limit, we automatically try the next one
    (that's what LiteLLM's fallback does for us)

Usage:
  python analyzer.py                        # interactive mode — paste your text
  python analyzer.py "some passage here"    # pass text directly
  python analyzer.py --file passage.txt     # read from a file

Author: Satish Premanand
Date: May 2026
"""

import os
import sys
import json
from dotenv import load_dotenv
from litellm import completion

# Grab our API key from .env so we don't hardcode secrets
load_dotenv()


# ─────────────────────────────────────────────────────────────────────
# MODEL FALLBACK CHAIN
# ─────────────────────────────────────────────────────────────────────
# We line up 4 models — if the first one is busy or rate-limited,
# LiteLLM automatically rolls over to the next. Think of it like
# having backup generators when the power goes out.

MODELS = [
    "groq/llama-3.1-8b-instant",                            # Primary: fast and reliable
    "groq/meta-llama/llama-4-scout-17b-16e-instruct",       # Fallback 1: newer, larger model
    "groq/gemma2-9b-it",                                    # Fallback 2: Google's Gemma, different architecture
    "groq/llama-guard-3-8b",                                # Fallback 3: last resort
]


# ─────────────────────────────────────────────────────────────────────
# WORD COUNT
# ─────────────────────────────────────────────────────────────────────
# No AI needed for this — just split on whitespace and count.

def count_words(text):
    """Count total words in the passage. Simple split-based approach."""
    words = text.split()
    return len(words)


# ─────────────────────────────────────────────────────────────────────
# LLM CALL WITH FALLBACK
# ─────────────────────────────────────────────────────────────────────
# This is the core function that talks to Groq. It tries each model
# in order until one responds. If ALL of them fail, we raise an error
# rather than silently returning garbage.

def call_llm(prompt, system_message="You are a helpful literary analyst."):
    """
    Send a prompt to Groq via LiteLLM, with automatic model fallback.
    
    Args:
        prompt: What we're asking the model
        system_message: Sets the model's persona/behavior
    
    Returns:
        The model's response text
    
    Raises:
        Exception if all models fail (unlikely but possible)
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]

    # Try each model in the chain
    for i, model in enumerate(MODELS):
        try:
            response = completion(
                model=model,
                messages=messages,
                temperature=0.3,  # Low temp = more consistent, less random
                max_tokens=1024,
            )
            return response.choices[0].message.content

        except Exception as e:
            # If this isn't the last model, log it and move on
            if i < len(MODELS) - 1:
                print(f"  ⚠ {model} unavailable ({str(e)[:50]}...), trying next model...")
            else:
                # We've exhausted all options
                raise Exception(f"All models failed. Last error: {e}")


# ─────────────────────────────────────────────────────────────────────
# EMOTION DETECTION
# ─────────────────────────────────────────────────────────────────────
# We ask the LLM to identify the dominant emotion and explain why.
# The JSON format keeps things parseable so we can display it nicely.

def detect_emotion(text):
    """
    Figure out what emotion dominates this passage.
    
    Returns a dict like:
        {"emotion": "melancholy", "confidence": "85%", "reasoning": "..."}
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

    # Try to parse the JSON from the response
    try:
        # Sometimes models wrap JSON in markdown code blocks — strip that
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # If parsing fails, return the raw text so we don't crash
        return {"emotion": "unknown", "confidence": "N/A", "reasoning": response}


# ─────────────────────────────────────────────────────────────────────
# BOOK IDENTIFICATION
# ─────────────────────────────────────────────────────────────────────
# This is where the LLM's training data really shines — it's read
# millions of books and can pattern-match writing style, themes, and
# vocabulary to guess the source.

def identify_books(text):
    """
    Guess 2-3 books this passage might be from.
    
    Returns a list like:
        [{"title": "...", "author": "...", "reason": "..."}]
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


# ─────────────────────────────────────────────────────────────────────
# SUMMARIZATION
# ─────────────────────────────────────────────────────────────────────
# Keep it short and sweet — 2-3 sentences that capture the essence.

def summarize_passage(text):
    """
    Produce a concise 2-3 sentence summary of the passage.
    
    Returns the summary as a plain string.
    """
    prompt = f"""Summarize this passage in 2-3 clear, concise sentences. 
Capture the main idea, tone, and any key events or themes.

Passage:
\"{text}\""""

    return call_llm(prompt)


# ─────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────
# This ties everything together — runs all analyses and prints a
# nicely formatted report. Also saves results to JSON for later use.

def analyze_passage(text):
    """
    Run the full analysis pipeline on a passage:
      1. Count words
      2. Detect emotion
      3. Identify possible books
      4. Summarize
    
    Prints formatted results and saves them to analysis_results.json
    """
    print("\n" + "═" * 66)
    print("                    📖 BOOK PASSAGE ANALYZER")
    print("═" * 66)

    # --- Step 1: Word Count (instant, no API needed) ---
    word_count = count_words(text)
    print(f"\n📊 WORD COUNT")
    print(f"   Total words: {word_count}")

    # --- Step 2: Emotion Detection ---
    print(f"\n🎭 EMOTION ANALYSIS")
    print(f"   Analyzing...")
    emotion = detect_emotion(text)
    print(f"   Emotion   : {emotion.get('emotion', 'N/A').title()}")
    print(f"   Confidence: {emotion.get('confidence', 'N/A')}")
    print(f"   Reasoning : {emotion.get('reasoning', 'N/A')}")

    # --- Step 3: Book Identification ---
    print(f"\n📚 POSSIBLE BOOKS")
    print(f"   Searching...")
    books = identify_books(text)
    for i, book in enumerate(books, 1):
        print(f"   {i}. \"{book.get('title', 'Unknown')}\" by {book.get('author', 'Unknown')}")
        print(f"      → {book.get('reason', 'N/A')}")

    # --- Step 4: Summary ---
    print(f"\n📝 SUMMARY")
    print(f"   Generating...")
    summary = summarize_passage(text)
    # Wrap long summaries nicely
    wrapped = summary.replace(". ", ".\n   ")
    print(f"   {wrapped}")

    print("\n" + "═" * 66)

    # --- Save everything to a JSON file for reference ---
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


# ─────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────
# Three ways to use this script:
#   1. Pass a passage as a command-line argument
#   2. Point it to a text file with --file
#   3. Just run it and paste your text interactively

if __name__ == "__main__":

    # Check that the API key is set up
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY not found!")
        print("   Create a .env file with: GROQ_API_KEY=your_key_here")
        print("   Get your key at: https://console.groq.com/keys")
        sys.exit(1)

    # Determine where the passage is coming from
    if len(sys.argv) > 1:
        if sys.argv[1] == "--file":
            # Reading from a file
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
            # Passage passed directly as argument
            passage = " ".join(sys.argv[1:])
    else:
        # Interactive mode — ask the user to paste text
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

    # Make sure we actually got some text
    if not passage.strip():
        print("❌ No text provided. Nothing to analyze!")
        sys.exit(1)

    # Run the analysis
    analyze_passage(passage)
