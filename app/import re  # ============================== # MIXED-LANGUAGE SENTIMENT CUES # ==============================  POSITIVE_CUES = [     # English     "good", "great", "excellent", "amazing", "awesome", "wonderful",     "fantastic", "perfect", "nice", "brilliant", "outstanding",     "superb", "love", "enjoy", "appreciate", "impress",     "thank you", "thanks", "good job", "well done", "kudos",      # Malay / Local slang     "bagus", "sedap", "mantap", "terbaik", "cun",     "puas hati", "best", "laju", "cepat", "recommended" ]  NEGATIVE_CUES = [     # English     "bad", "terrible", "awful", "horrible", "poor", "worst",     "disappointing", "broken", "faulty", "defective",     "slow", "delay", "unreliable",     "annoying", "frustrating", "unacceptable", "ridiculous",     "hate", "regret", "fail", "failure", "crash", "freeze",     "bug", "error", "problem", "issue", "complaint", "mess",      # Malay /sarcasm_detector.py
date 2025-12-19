import re

# ==============================
# MIXED-LANGUAGE SENTIMENT CUES
# ==============================

POSITIVE_CUES = [
    # English
    "good", "great", "excellent", "amazing", "awesome", "wonderful",
    "fantastic", "perfect", "nice", "brilliant", "outstanding",
    "superb", "love", "enjoy", "appreciate", "impress",
    "thank you", "thanks", "good job", "well done", "kudos",

    # Malay / Local slang
    "bagus", "sedap", "mantap", "terbaik", "cun",
    "puas hati", "best", "laju", "cepat", "recommended"
]

NEGATIVE_CUES = [
    # English
    "bad", "terrible", "awful", "horrible", "poor", "worst",
    "disappointing", "broken", "faulty", "defective",
    "slow", "delay", "unreliable",
    "annoying", "frustrating", "unacceptable", "ridiculous",
    "hate", "regret", "fail", "failure", "crash", "freeze",
    "bug", "error", "problem", "issue", "complaint", "mess",

    # Malay / Local slang
    "lambat", "teruk", "rosak", "cacat", "tak function",
    "tak puas hati", "mengecewakan", "marah", "sakit hati",
    "rugi", "bazir", "hancur", "susah", "leceh"
]

# ==============================
# SEGMENT-LEVEL PROCESSING
# ==============================

def split_into_segments(text):
    return [
        seg.strip()
        for seg in re.split(r"[,.!;]| but | tapi | however | walaupun ", text.lower())
        if seg.strip()
    ]

def contains_positive(segment):
    return any(p in segment for p in POSITIVE_CUES)

def contains_negative(segment):
    return any(n in segment for n in NEGATIVE_CUES)

# ==============================
# HYBRID SARCASM DETECTION
# ==============================

def detect_sarcastic_hybrid(review_text):
    segments = split_into_segments(review_text)

    positive_found = False
    negative_found = False

    for seg in segments:
        if contains_positive(seg):
            positive_found = True
        if contains_negative(seg):
            negative_found = True

    # Sentiment incongruity → sarcasm
    if positive_found and negative_found:
        return True

    return False

def detect_sarcasm_with_confidence(review_text, sentiment_confidence):
    contradiction = detect_sarcastic_hybrid(review_text)

    # Low confidence + contradiction = sarcasm
    if contradiction and sentiment_confidence < 0.65:
        return True

    return False
