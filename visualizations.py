"""
Figures for the text anonymisation project.

Every value plotted is computed by evaluation.py or conll_evaluation.py at
run time. Nothing is hardcoded.

Note: the previous version of this file plotted a precision-recall curve from
a hardcoded list of made-up coordinates while claiming all numbers were real.
That panel has been removed rather than fixed: the pipeline emits hard
decisions, not ranked scores, so there is no threshold to sweep and no
precision-recall curve to draw.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation import run_evaluation_demo, run_spacy_evaluation
from conll_evaluation import run_conll_evaluation

BERT_COLOUR = "#2e7d5b"
SPACY_COLOUR = "#9dc3b4"


def figure_one(bert, spacy_scorer):
    """Two panels: overall comparison, and per-entity F1 for the pipeline."""
    plt.rcParams.update({"font.size": 9, "font.family": "serif"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.15))

    bo, so = bert.overall(), spacy_scorer.overall()
    labels = ["Precision", "Recall", "F1"]
    b = [bo["precision"] * 100, bo["recall"] * 100, bo["f1"] * 100]
    s = [so["precision"] * 100, so["recall"] * 100, so["f1"] * 100]

    x, w = np.arange(3), 0.34
    r1 = ax1.bar(x - w / 2, b, w, label="BERT + Regex", color=BERT_COLOUR)
    r2 = ax1.bar(x + w / 2, s, w, label="spaCy baseline", color=SPACY_COLOUR)
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 128); ax1.set_yticks([0, 25, 50, 75, 100])
    ax1.set_ylabel("Score (%)")
    ax1.set_title("Overall: pipeline vs. spaCy", fontsize=9.5, pad=4)
    ax1.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.02), handlelength=1.2, columnspacing=1.0)
    for bars in (r1, r2):
        ax1.bar_label(bars, fmt="%.0f%%", padding=1.5, fontsize=7.5)
    ax1.spines[["top", "right"]].set_visible(False)

    per = bert.per_entity()
    ents = [e for e in per if per[e]["tp"] + per[e]["fn"] > 0]
    f1 = [per[e]["f1"] * 100 for e in ents]
    y = np.arange(len(ents))[::-1]
    bars = ax2.barh(y, f1, 0.62, color=BERT_COLOUR)
    ax2.set_yticks(y); ax2.set_yticklabels(ents, fontsize=8)
    ax2.set_xlim(0, 118); ax2.set_xticks([0, 25, 50, 75, 100])
    ax2.set_xlabel("F1 (%)")
    ax2.set_title("Per-entity F1, full pipeline", fontsize=9.5, pad=4)
    ax2.bar_label(bars, fmt="%.0f%%", padding=2.5, fontsize=7.5)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout(pad=0.3, w_pad=1.8)
    plt.savefig("anonymization_results.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved anonymization_results.png")


def figure_two(bert, spacy_scorer, entity_types):
    plt.rcParams.update({"font.size": 9, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    b = [bert.per_entity()[e]["f1"] * 100 for e in entity_types]
    s = [spacy_scorer.per_entity()[e]["f1"] * 100 for e in entity_types]
    x, w = np.arange(len(entity_types)), 0.35
    r1 = ax.bar(x - w / 2, b, w, label="BERT NER", color=BERT_COLOUR)
    r2 = ax.bar(x + w / 2, s, w, label="spaCy", color=SPACY_COLOUR)
    ax.set_xticks(x); ax.set_xticklabels(entity_types, fontsize=8)
    ax.set_ylim(0, 122); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("F1 (%)")
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, 1.04))
    for bars in (r1, r2):
        ax.bar_label(bars, fmt="%.0f%%", padding=1.5, fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(pad=0.3)
    plt.savefig("conll_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved conll_comparison.png")


def main():
    bert, _, _ = run_evaluation_demo(verbose=False)
    spacy_scorer = run_spacy_evaluation()
    figure_one(bert, spacy_scorer)

    conll_bert, conll_spacy = run_conll_evaluation()
    figure_two(conll_bert, conll_spacy, ["NAME", "LOCATION", "ORGANIZATION"])

    o = bert.overall()
    print("\nNumbers to copy into the paper")
    print("-" * 40)
    print(f"Overall (micro): P {o['precision']:.0%}  R {o['recall']:.0%}  F1 {o['f1']:.0%}")
    for etype, v in bert.per_entity().items():
        if v["tp"] + v["fn"]:
            print(f"  {etype:<14} {v['precision']:.0%} / {v['recall']:.0%} / {v['f1']:.0%}")


if __name__ == "__main__":
    main()