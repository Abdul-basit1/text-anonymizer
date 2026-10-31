"""
CoNLL-2003 style evaluation: BERT + regex pipeline against the spaCy baseline
on the three entity types both systems support.
"""

from collections import Counter

from anonymizer import TextAnonymizer
from evaluation import EntityScorer, spacy_predictions
import spacy

CONLL_TYPES = ["NAME", "LOCATION", "ORGANIZATION"]

CONLL_TEST_SAMPLES = [
    {"text": "Peter Blackburn works at BRUSSELS for the European Commission.",
     "entities": [{"word": "Peter Blackburn", "entity": "NAME"},
                  {"word": "BRUSSELS", "entity": "LOCATION"},
                  {"word": "European Commission", "entity": "ORGANIZATION"}]},
    {"text": "John Smith traveled from London to New York last Monday.",
     "entities": [{"word": "John Smith", "entity": "NAME"},
                  {"word": "London", "entity": "LOCATION"},
                  {"word": "New York", "entity": "LOCATION"}]},
    {"text": "Microsoft CEO Satya Nadella announced the partnership in Seattle.",
     "entities": [{"word": "Microsoft", "entity": "ORGANIZATION"},
                  {"word": "Satya Nadella", "entity": "NAME"},
                  {"word": "Seattle", "entity": "LOCATION"}]},
    {"text": "Dr. Angela Merkel met President Emmanuel Macron in Paris.",
     "entities": [{"word": "Angela Merkel", "entity": "NAME"},
                  {"word": "Emmanuel Macron", "entity": "NAME"},
                  {"word": "Paris", "entity": "LOCATION"}]},
    {"text": "The United Nations headquarters is located in New York City.",
     "entities": [{"word": "United Nations", "entity": "ORGANIZATION"},
                  {"word": "New York City", "entity": "LOCATION"}]},
    {"text": "Elon Musk founded SpaceX in California.",
     "entities": [{"word": "Elon Musk", "entity": "NAME"},
                  {"word": "SpaceX", "entity": "ORGANIZATION"},
                  {"word": "California", "entity": "LOCATION"}]},
    {"text": "Professor James Watson works at Harvard University in Cambridge.",
     "entities": [{"word": "James Watson", "entity": "NAME"},
                  {"word": "Harvard University", "entity": "ORGANIZATION"},
                  {"word": "Cambridge", "entity": "LOCATION"}]},
    {"text": "Tokyo hosted the Olympics, organized by the Japanese Olympic Committee.",
     "entities": [{"word": "Tokyo", "entity": "LOCATION"},
                  {"word": "Japanese Olympic Committee", "entity": "ORGANIZATION"}]},
    {"text": "Mark Zuckerberg launched Facebook from his dorm room in Boston.",
     "entities": [{"word": "Mark Zuckerberg", "entity": "NAME"},
                  {"word": "Facebook", "entity": "ORGANIZATION"},
                  {"word": "Boston", "entity": "LOCATION"}]},
    {"text": "The report was written by Sarah Connor in Los Angeles.",
     "entities": [{"word": "Sarah Connor", "entity": "NAME"},
                  {"word": "Los Angeles", "entity": "LOCATION"}]},
]


def check_ground_truth():
    """Every annotated surface form must actually occur in its sentence."""
    problems = []
    for s in CONLL_TEST_SAMPLES:
        for e in s["entities"]:
            if e["word"] not in s["text"]:
                problems.append((s["text"], e["word"]))
    return problems


def dataset_statistics():
    counts = Counter(e["entity"] for s in CONLL_TEST_SAMPLES for e in s["entities"])
    total = sum(counts.values())
    print("\nDATASET STATISTICS")
    print("-" * 40)
    print(f"Samples:  {len(CONLL_TEST_SAMPLES)}")
    print(f"Entities: {total}")
    for etype, n in counts.most_common():
        print(f"  {etype}: {n} ({n / total * 100:.1f}%)")
    return counts


def run_conll_evaluation():
    print("\n" + "=" * 70)
    print("CoNLL-2003 STYLE EVALUATION")
    print("=" * 70)

    problems = check_ground_truth()
    if problems:
        raise ValueError(f"ground-truth strings not found in their sentences: {problems}")

    dataset_statistics()

    anonymizer = TextAnonymizer()
    nlp = spacy.load("en_core_web_sm")

    bert = EntityScorer(CONLL_TYPES)
    spacy_scorer = EntityScorer(CONLL_TYPES)

    for sample in CONLL_TEST_SAMPLES:
        text, gold = sample["text"], sample["entities"]

        _, preds = anonymizer.anonymize(text)
        preds = [p for p in preds if p["entity"] in CONLL_TYPES]
        bert.add(preds, gold)

        sp = [p for p in spacy_predictions(nlp, text) if p["entity"] in CONLL_TYPES]
        spacy_scorer.add(sp, gold)

    print(bert.table("BERT + REGEX"))
    print(spacy_scorer.table("spaCy en_core_web_sm"))

    print("\n" + "=" * 46)
    print(f"{'Entity':<16}{'BERT F1':>14}{'spaCy F1':>14}")
    print("-" * 46)
    b, s = bert.per_entity(), spacy_scorer.per_entity()
    for etype in CONLL_TYPES:
        print(f"{etype:<16}{b[etype]['f1']:>13.1%}{s[etype]['f1']:>14.1%}")
    bo, so = bert.overall(), spacy_scorer.overall()
    print("-" * 46)
    print(f"{'OVERALL':<16}{bo['f1']:>13.1%}{so['f1']:>14.1%}")

    return bert, spacy_scorer


if __name__ == "__main__":
    run_conll_evaluation()