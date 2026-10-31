"""
Automatic Text Anonymization using NLP
Hybrid pipeline: BERT NER (context-dependent entities) + regex (structured entities).
"""

import re
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)

# Higher number wins when two detections overlap.
# Regex detections are more reliable than the model on structured entities,
# so they outrank it.
PRIORITY = {
    "EMAIL": 3,
    "DATE": 2,
    "NAME": 1,
    "LOCATION": 1,
    "ORGANIZATION": 1,
}

NER_LABEL_MAP = {
    "PER": "NAME", "PERSON": "NAME",
    "LOC": "LOCATION", "LOCATION": "LOCATION", "GPE": "LOCATION",
    "ORG": "ORGANIZATION", "ORGANIZATION": "ORGANIZATION",
}


def _overlaps(a, b):
    """True if two half-open character spans intersect."""
    return not (a["end"] <= b["start"] or a["start"] >= b["end"])


class TextAnonymizer:
    def __init__(self, model_name="dslim/bert-base-NER", min_score=0.50):
        print("Loading NER model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.ner_pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
        )
        self.min_score = min_score
        print("Model loaded successfully!")

        self.email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        # Ordered by specificity: the four-digit-year format is tested first so
        # that a shorter pattern cannot consume part of a longer date.
        self.date_patterns = [
            r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
            r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?,?\s+\d{2,4}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4}\b',
        ]

    # ---------- detectors ----------

    def detect_emails(self, text):
        return [
            {"word": m.group(), "start": m.start(), "end": m.end(), "entity": "EMAIL"}
            for m in re.finditer(self.email_pattern, text, re.IGNORECASE)
        ]

    def detect_dates(self, text):
        """Patterns are tried in specificity order; an earlier match locks its span."""
        results, taken = [], []
        for pattern in self.date_patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                cand = {"word": m.group(), "start": m.start(), "end": m.end(),
                        "entity": "DATE"}
                if not any(_overlaps(cand, t) for t in taken):
                    results.append(cand)
                    taken.append(cand)
        return results

    def detect_ner_entities(self, text):
        results = []
        for ent in self.ner_pipeline(text):
            label = NER_LABEL_MAP.get(ent["entity_group"])
            if label is None:
                continue
            if float(ent.get("score", 1.0)) < self.min_score:
                continue
            start, end = int(ent["start"]), int(ent["end"])
            # aggregation_strategy="simple" can include leading/trailing
            # whitespace or punctuation in the span; trim it so the mask
            # replaces the entity and nothing else.
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1] in " \t\n.,;:!?":
                end -= 1
            if start >= end:
                continue
            results.append({"word": text[start:end], "start": start, "end": end,
                            "entity": label})
        return results

    # ---------- merge ----------

    def resolve_overlaps(self, entities):
        """
        Keep the higher-priority detection when spans conflict.

        Ordering rule: priority first (regex beats the model), then the longer
        span, then position. Sorting by position alone lets a NAME detected
        inside an email address win, which masks the local part and leaves the
        domain in the output.
        """
        ordered = sorted(
            entities,
            key=lambda e: (-PRIORITY.get(e["entity"], 0),
                           -(e["end"] - e["start"]),
                           e["start"]),
        )
        kept = []
        for ent in ordered:
            if not any(_overlaps(ent, k) for k in kept):
                kept.append(ent)
        return sorted(kept, key=lambda e: e["start"])

    def anonymize(self, text):
        entities = (
            self.detect_ner_entities(text)
            + self.detect_emails(text)
            + self.detect_dates(text)
        )
        kept = self.resolve_overlaps(entities)

        # Replace right-to-left so earlier offsets stay valid.
        out = text
        for ent in reversed(kept):
            out = out[:ent["start"]] + f"[{ent['entity']}]" + out[ent["end"]:]
        return out, kept


def main():
    anonymizer = TextAnonymizer()

    test_texts = [
        "John Smith lives in Berlin and was born on 12 May 1995. His email is john.smith@gmail.com.",
        "Please contact Sarah Johnson at sarah.j@company.org. She works in New York.",
        "The meeting with Dr. Michael Brown is scheduled for 2024-03-15 in London.",
        "Send the report to alex.kumar@outlook.com by 25/12/2024. Alex Kumar from Paris will review it.",
    ]

    print("\n" + "=" * 60)
    print("TEXT ANONYMIZATION DEMO")
    print("=" * 60)

    for i, text in enumerate(test_texts, 1):
        anonymized, entities = anonymizer.anonymize(text)
        print(f"\n--- Example {i} ---")
        print(f"INPUT:  {text}")
        print(f"OUTPUT: {anonymized}")
        print(f"ENTITIES: {[(e['word'], e['entity']) for e in entities]}")


if __name__ == "__main__":
    main()