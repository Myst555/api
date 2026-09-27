import re
from collections import Counter
from typing import Dict, List, Tuple


def _normalize_words(text: str) -> List[str]:
    cleaned = re.sub(r"[^a-zA-Z0-9' ]+", " ", text.lower())
    return [word for word in cleaned.split() if word]


def _compute_base_metrics(reference_text: str, recognized_text: str) -> Tuple[dict, List[str], List[str]]:
    reference_words = _normalize_words(reference_text)
    recognized_words = _normalize_words(recognized_text)

    reference_counter = Counter(reference_words)
    recognized_counter = Counter(recognized_words)
    matched_counter = reference_counter & recognized_counter

    matched_words = sum(matched_counter.values())
    reference_word_count = len(reference_words)
    accuracy_percent = (
        (matched_words / reference_word_count) * 100 if reference_word_count else 0.0
    )

    missing_counter = reference_counter - recognized_counter
    # Keep weak words unique and ordered by missing frequency.
    weak_words = [word for word, _ in missing_counter.most_common(10)]

    if reference_word_count == 0:
        feedback = "No reference text provided; transcription only."
    elif accuracy_percent >= 85:
        feedback = "Great reading clarity."
    elif accuracy_percent >= 60:
        feedback = "Good attempt. Practice missed words and pacing."
    else:
        feedback = "Try slower reading and clearer pronunciation."

    return (
        {
            "accuracy_percent": round(accuracy_percent, 2),
            "matched_words": matched_words,
            "reference_word_count": reference_word_count,
            "weak_words": weak_words,
            "feedback": feedback,
        },
        reference_words,
        recognized_words,
    )


def analyze_pronunciation(
    reference_text: str,
    recognized_text: str,
    audio_path: str = "",
) -> Dict:
    base_metrics, _, _ = _compute_base_metrics(
        reference_text,
        recognized_text,
    )

    return {
        **base_metrics,
        "phoneme_mode": "disabled_whisper_only",
        "phoneme_accuracy_percent": None,
        "phoneme_edit_distance": None,
        "decoded_phoneme_count": 0,
        "decoded_phonemes_preview": [],
        "phoneme_debug": "phoneme_analysis_removed",
    }