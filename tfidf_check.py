# tfidf_check.py
import os
import re
from math import floor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
nltk.download("punkt", quiet=True)

CORPUS_DIR = "corpus"
WINDOW_SIZE = 3           # number of sentences per window
SIM_THRESHOLD = 0.5       # similarity threshold for a window to be considered matched


def load_corpus():
    names = []
    docs = []
    for fname in os.listdir(CORPUS_DIR):
        path = os.path.join(CORPUS_DIR, fname)
        if os.path.isfile(path) and fname.lower().endswith(".txt"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                names.append(fname)
                docs.append(f.read())
    return names, docs


def make_windows_from_text(text, window_size=WINDOW_SIZE):
    sents = sent_tokenize(text)
    windows = []
    for i in range(max(1, len(sents) - window_size + 1)):
        windows.append(" ".join(sents[i:i+window_size]))
    if not windows:
        windows.append(text)
    return windows


def compute_matches_and_highlight(query_text, corpus_names, corpus_texts):
    # Build TF-IDF vectorizer on corpus + query windows (we will transform per-window)
    # For speed and simplicity, vectorize corpus documents (doc-level) and query windows separately
    vectorizer = TfidfVectorizer(stop_words="english")
    # Fit on corpus docs to get a vocabulary (so query windows use same vectorizer)
    if not corpus_texts:
        return [], 0.0, query_text
    vectorizer.fit(corpus_texts)

    # Corpus vectors (doc-level)
    corpus_vecs = vectorizer.transform(corpus_texts)

    # Query windows
    q_windows = make_windows_from_text(query_text, window_size=WINDOW_SIZE)

    matches = []
    matched_windows = []  # store exact window texts that were matched
    for win in q_windows:
        qv = vectorizer.transform([win])
        sims = cosine_similarity(qv, corpus_vecs)[0]
        best_idx = int(sims.argmax())
        best_score = float(sims[best_idx])

        if best_score >= SIM_THRESHOLD:
            matches.append({
                "matched_file": corpus_names[best_idx],
                "score": round(best_score, 3),
                "query_window": win,
                "matched_text_snippet": corpus_texts[best_idx][:400]
            })
            matched_windows.append(win)

    # Compute overall percentage: count words in matched windows (unique occurrences) vs total words
    total_words = len(word_tokenize(query_text))
    # To avoid double counting overlapping windows, concatenate matched windows and count unique word spans
    matched_text_combined = " ".join(matched_windows)
    matched_words = len(word_tokenize(matched_text_combined))
    overall_percent = round((matched_words / total_words) * 100, 2) if total_words > 0 else 0.0

    # Highlight matched windows in the original query text
    highlighted = query_text
    # Replace occurrences of each matched window with a marked version.
    # Use re.escape and replace only the first occurrence to avoid over-highlighting.
    for mw in sorted(set(matched_windows), key=lambda x: -len(x)):  # longest first
        try:
            pattern = re.escape(mw)
            # Replace the first occurrence only (case-sensitive). Use a function to keep original.
            highlighted, n = re.subn(pattern, f"<mark>{mw}</mark>", highlighted, count=1)
        except re.error:
            # fallback to simple replace
            highlighted = highlighted.replace(mw, f"<mark>{mw}</mark>", 1)

    return matches, overall_percent, highlighted


def check_text(suspicious_text):
    names, docs = load_corpus()
    matches, overall_percent, highlighted_html = compute_matches_and_highlight(suspicious_text, names, docs)
    # Return structured results for frontend
    return {
        "matches": matches,
        "overall_percent": overall_percent,
        "highlighted_html": highlighted_html
    }


# Optional CLI test
if __name__ == "__main__":
    q = input("Paste suspicious text:\n")
    r = check_text(q)
    print("Overall percent:", r["overall_percent"])
    for m in r["matches"]:
        print(m)
