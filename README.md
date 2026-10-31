# Text Anonymizer

Automatic text anonymization using NLP. The pipeline finds personal information in text and replaces it with placeholders:

```
INPUT:  John Smith lives in Berlin and was born on 12 May 1995. His email is john.smith@gmail.com.
OUTPUT: [NAME] lives in [LOCATION] and was born on [DATE]. His email is [EMAIL].
```

## How it works

It is a hybrid pipeline:

- **BERT NER** ([`dslim/bert-base-NER`](https://huggingface.co/dslim/bert-base-NER)) detects context-dependent entities: names, locations and organizations.
- **Regular expressions** detect structured entities: email addresses and dates in several formats (`2024-03-15`, `25/12/2024`, `12 May 1995`, `May 12, 1995`).
- When detections overlap, the higher-priority one is kept (email > date > model entities), then the longer span.

Entity types: `NAME`, `LOCATION`, `ORGANIZATION`, `DATE`, `EMAIL`.

## Files

| File | Description |
|---|---|
| `anonymizer.py` | The `TextAnonymizer` class and a demo on sample sentences |
| `evaluation.py` | Precision, recall and F1 per entity type on hand-labelled test sentences, compared against a spaCy (`en_core_web_sm`) baseline |
| `conll_evaluation.py` | CoNLL-2003-style comparison of the pipeline and spaCy on names, locations and organizations, using a small hand-written sample set |
| `visualizations.py` | Runs both evaluations and saves the result charts as PNG files |
| `SETUP.txt` | Detailed setup instructions for Windows, Linux and macOS, plus troubleshooting |
| `PostProjectPaper_AbdulBasit_abdba332.pdf` | Project paper |
| `*.png` | Result charts used in the paper |

## Quick start

```sh
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt matplotlib
python -m spacy download en_core_web_sm

python anonymizer.py                # demo
python evaluation.py                # evaluation + spaCy baseline
python conll_evaluation.py          # CoNLL-style comparison
python visualizations.py            # generate charts
```

The first run downloads the BERT model (about 400 MB). See `SETUP.txt` for platform-specific notes.

## Usage

```python
from anonymizer import TextAnonymizer

anonymizer = TextAnonymizer()
text, entities = anonymizer.anonymize("Contact Sarah at sarah@test.com. She is in Paris.")
print(text)  # Contact [NAME] at [EMAIL]. She is in [LOCATION].
```

## Note

The evaluation sets are small (17 and 10 sentences), so the scores are indicative rather than benchmark results.
