"""
Evaluation for the text anonymisation system.

Per-entity and overall metrics come from a single accumulator (EntityScorer
below), so the overall row is the micro-sum of the per-entity rows and cannot
disagree with them.

Matching is multiset-based. Using a set of (word, type) pairs silently
collapses repeated entities, both within a sentence and, if the sets are
built over the whole corpus, across sentences.
"""

from collections import Counter

from anonymizer import TextAnonymizer
import spacy


class EntityScorer:
    def __init__(self, entity_types):
        self.entity_types = list(entity_types)
        self.counts = {e: {"tp": 0, "fp": 0, "fn": 0} for e in self.entity_types}

    @staticmethod
    def _bag(entities):
        return Counter((e["word"].lower().strip(), e["entity"]) for e in entities)

    def add(self, predictions, ground_truth):
        """Score one sentence and fold the result into the running totals."""
        pred, true = self._bag(predictions), self._bag(ground_truth)
        matched = pred & true
        for (_, etype), n in matched.items():
            if etype in self.counts:
                self.counts[etype]["tp"] += n
        for (_, etype), n in (pred - true).items():
            if etype in self.counts:
                self.counts[etype]["fp"] += n
        for (_, etype), n in (true - pred).items():
            if etype in self.counts:
                self.counts[etype]["fn"] += n

    @staticmethod
    def _prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f

    def per_entity(self):
        out = {}
        for etype, c in self.counts.items():
            p, r, f = self._prf(c["tp"], c["fp"], c["fn"])
            out[etype] = {"precision": p, "recall": r, "f1": f, **c}
        return out

    def overall(self):
        """Micro-average: sum the counts, then compute the metrics once."""
        tp = sum(c["tp"] for c in self.counts.values())
        fp = sum(c["fp"] for c in self.counts.values())
        fn = sum(c["fn"] for c in self.counts.values())
        p, r, f = self._prf(tp, fp, fn)
        return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}

    def macro(self):
        live = [v for v in self.per_entity().values() if v["tp"] + v["fn"] > 0]
        if not live:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        n = len(live)
        return {k: sum(v[k] for v in live) / n for k in ("precision", "recall", "f1")}

    def table(self, title="RESULTS"):
        lines = ["", "=" * 60, title, "=" * 60,
                 f"{'Entity':<15}{'P':>10}{'R':>10}{'F1':>10}{'TP':>6}{'FP':>5}{'FN':>5}",
                 "-" * 60]
        for etype, v in self.per_entity().items():
            if v["tp"] + v["fp"] + v["fn"] == 0:
                continue
            lines.append(f"{etype:<15}{v['precision']:>9.1%}{v['recall']:>10.1%}"
                         f"{v['f1']:>10.1%}{v['tp']:>6}{v['fp']:>5}{v['fn']:>5}")
        o, m = self.overall(), self.macro()
        lines += ["-" * 60,
                  f"{'OVERALL (micro)':<15}{o['precision']:>9.1%}{o['recall']:>10.1%}"
                  f"{o['f1']:>10.1%}{o['tp']:>6}{o['fp']:>5}{o['fn']:>5}",
                  f"{'OVERALL (macro)':<15}{m['precision']:>9.1%}{m['recall']:>10.1%}"
                  f"{m['f1']:>10.1%}"]
        return "\n".join(lines)


ENTITY_TYPES = ["NAME", "LOCATION", "DATE", "EMAIL", "ORGANIZATION"]

SPACY_LABEL_MAP = {
    "PERSON": "NAME",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "ORG": "ORGANIZATION",
    "DATE": "DATE",
}

TEST_CASES = [
    {"text": "John Smith lives in Berlin and was born on 12 May 1995. His email is john.smith@gmail.com.",
     "ground_truth": [{"word": "John Smith", "entity": "NAME"},
                      {"word": "Berlin", "entity": "LOCATION"},
                      {"word": "12 May 1995", "entity": "DATE"},
                      {"word": "john.smith@gmail.com", "entity": "EMAIL"}]},
    {"text": "Contact Sarah at sarah@test.com. She is in Paris.",
     "ground_truth": [{"word": "Sarah", "entity": "NAME"},
                      {"word": "sarah@test.com", "entity": "EMAIL"},
                      {"word": "Paris", "entity": "LOCATION"}]},
    {"text": "Meeting with Dr. Brown on 2024-01-15 in London office.",
     "ground_truth": [{"word": "Dr. Brown", "entity": "NAME"},
                      {"word": "2024-01-15", "entity": "DATE"},
                      {"word": "London", "entity": "LOCATION"}]},
    {"text": "Please contact Sarah Johnson at sarah.j@company.org. She works in New York.",
     "ground_truth": [{"word": "Sarah Johnson", "entity": "NAME"},
                      {"word": "sarah.j@company.org", "entity": "EMAIL"},
                      {"word": "New York", "entity": "LOCATION"}]},
    {"text": "Send the report to alex.kumar@outlook.com by 25/12/2024. Alex Kumar from Paris will review it.",
     "ground_truth": [{"word": "alex.kumar@outlook.com", "entity": "EMAIL"},
                      {"word": "25/12/2024", "entity": "DATE"},
                      {"word": "Alex Kumar", "entity": "NAME"},
                      {"word": "Paris", "entity": "LOCATION"}]},
    {"text": "Emma Wilson at emma.w@gmail.com was born on 01 Jan 1990 and lives in Stockholm.",
     "ground_truth": [{"word": "Emma Wilson", "entity": "NAME"},
                      {"word": "emma.w@gmail.com", "entity": "EMAIL"},
                      {"word": "01 Jan 1990", "entity": "DATE"},
                      {"word": "Stockholm", "entity": "LOCATION"}]},
    {"text": "David Lee is based in Tokyo. Contact him at david.lee@test.com.",
     "ground_truth": [{"word": "David Lee", "entity": "NAME"},
                      {"word": "Tokyo", "entity": "LOCATION"},
                      {"word": "david.lee@test.com", "entity": "EMAIL"}]},
    {"text": "The contract was signed on 2023-11-20 in Amsterdam.",
     "ground_truth": [{"word": "2023-11-20", "entity": "DATE"},
                      {"word": "Amsterdam", "entity": "LOCATION"}]},
    {"text": "Robert Brown was born on 15/08/1988 and currently lives in Dublin.",
     "ground_truth": [{"word": "Robert Brown", "entity": "NAME"},
                      {"word": "15/08/1988", "entity": "DATE"},
                      {"word": "Dublin", "entity": "LOCATION"}]},
    {"text": "Anna Schmidt submitted her application on 2024-07-01 to the office in Vienna.",
     "ground_truth": [{"word": "Anna Schmidt", "entity": "NAME"},
                      {"word": "2024-07-01", "entity": "DATE"},
                      {"word": "Vienna", "entity": "LOCATION"}]},
    {"text": "She works at Google in Mountain View.",
     "ground_truth": [{"word": "Google", "entity": "ORGANIZATION"},
                      {"word": "Mountain View", "entity": "LOCATION"}]},
    {"text": "Microsoft announced a partnership with OpenAI on 2024-01-10.",
     "ground_truth": [{"word": "Microsoft", "entity": "ORGANIZATION"},
                      {"word": "OpenAI", "entity": "ORGANIZATION"},
                      {"word": "2024-01-10", "entity": "DATE"}]},
    {"text": "The World Bank published its annual economic report.",
     "ground_truth": [{"word": "World Bank", "entity": "ORGANIZATION"}]},
    {"text": "Air France flights were delayed on 2024-05-20.",
     "ground_truth": [{"word": "Air France", "entity": "ORGANIZATION"},
                      {"word": "2024-05-20", "entity": "DATE"}]},
    {"text": "The deal with Berkshire Hathaway fell through.",
     "ground_truth": [{"word": "Berkshire Hathaway", "entity": "ORGANIZATION"}]},
    {"text": "She completed her certification through edX.",
     "ground_truth": [{"word": "edX", "entity": "ORGANIZATION"}]},
    {"text": "Jordan submitted the report to the Geneva office.",
     "ground_truth": [{"word": "Jordan", "entity": "NAME"},
                      {"word": "Geneva", "entity": "LOCATION"}]},
]


def spacy_predictions(nlp, text):
    preds = []
    for ent in nlp(text).ents:
        mapped = SPACY_LABEL_MAP.get(ent.label_)
        if mapped:
            preds.append({"word": ent.text, "entity": mapped})
    return preds


def run_spacy_evaluation(test_cases=None):
    test_cases = test_cases or TEST_CASES
    nlp = spacy.load("en_core_web_sm")
    scorer = EntityScorer(ENTITY_TYPES)
    for case in test_cases:
        scorer.add(spacy_predictions(nlp, case["text"]), case["ground_truth"])
    return scorer


def run_evaluation_demo(verbose=True):
    anonymizer = TextAnonymizer()
    scorer = EntityScorer(ENTITY_TYPES)

    for i, case in enumerate(TEST_CASES, 1):
        _, predictions = anonymizer.anonymize(case["text"])
        scorer.add(predictions, case["ground_truth"])
        if verbose:
            pred = sorted((p["word"], p["entity"]) for p in predictions)
            true = sorted((t["word"], t["entity"]) for t in case["ground_truth"])
            flag = "" if pred == true else "   <-- mismatch"
            print(f"\n--- Test Case {i} ---{flag}")
            print(f"Text: {case['text'][:60]}...")
            print(f"  pred: {pred}")
            print(f"  gold: {true}")

    print(scorer.table("BERT + REGEX PIPELINE"))

    entity_counts = {e: 0 for e in ENTITY_TYPES}
    for case in TEST_CASES:
        for gt in case["ground_truth"]:
            entity_counts[gt["entity"]] += 1

    return scorer, entity_counts, TEST_CASES


if __name__ == "__main__":
    bert = run_evaluation_demo()[0]
    print(run_spacy_evaluation().table("spaCy BASELINE"))
