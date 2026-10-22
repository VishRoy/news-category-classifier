# News Category Classifier

A machine learning project that reads a news headline and predicts the article's category (for example **Politics**, **Sports**, **Business**, **Science & Tech** or **Entertainment**) with a confidence score.

> **Status:** in progress. The data has been explored and cleaned, and a TF-IDF + Logistic Regression model (balanced class weights) is trained and saved. A Streamlit app serves predictions with confidence scores, and predictions and user feedback are logged to SQLite. Next: a FastAPI `/predict` endpoint. The [Build log](#build-log) records each step and the reasoning behind it.

---

## Overview

News sites publish thousands of articles a day. Tagging each one with the right category by hand is slow and inconsistent. This project trains a text classifier that does the tagging automatically.

**Approach**

1. **Baseline:** TF-IDF features + Logistic Regression, in a single scikit-learn `Pipeline`
2. **Later:** a fine-tuned transformer (DistilBERT), compared against the baseline
3. Models are compared on accuracy, **macro-F1** and per-class recall (the classes are imbalanced, so accuracy alone is misleading)

## Dataset

[News Category Dataset v3](https://www.kaggle.com/datasets/rmisra/news-category-dataset): about 210k HuffPost headlines from 2012–2022, labelled with 42 categories.

| Field               | Description                         | Used?                            |
|---------------------|-------------------------------------|----------------------------------|
| `headline`          | Article headline                    | Yes, as the model input (`text`) |
| `short_description` | One-to-two sentence summary         | Not yet (planned)                |
| `category`          | Original label (42 values)          | Mapped to 16 labels (`label`)    |
| `link`, `authors`, `date` | Metadata                      | No                               |

Download the file from Kaggle and put it at `data/News_Category_Dataset_v3.json`. The data is **not** committed to git, because it is about 85 MB.

## Project Structure

```
news-category-classifier/
├── data/                  # downloaded dataset (not committed)
├── explore.py             # Step 1: data exploration
├── train.py               # Steps 2–3: cleaning, split, training, evaluation
├── app.py                 # Step 4: Streamlit UI (prediction, confidence, top 3, feedback)
├── db.py                  # Step 6: SQLite logging of predictions and feedback
├── results.md             # evaluation report for every training run (appended automatically)
├── sample_data.csv        # 50-row toy dataset used for the first prototype
├── requirements.txt
└── README.md
```

`model.joblib` (the saved model) and `predictions.db` (the prediction log) are created at runtime and aren't committed. Planned: a FastAPI service and a Dockerfile.

## Getting Started

```bash
git clone https://github.com/VishRoy/news-category-classifier.git
cd news-category-classifier

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Then download the dataset into `data/` (see [Dataset](#dataset)) and run:

```bash
python explore.py   # look at the data
python train.py     # clean, split, train, save model.joblib and append to results.md
streamlit run app.py   # open the web app at http://localhost:8501
```

## Results

Test set: 35,831 headlines (a stratified 20% split), 16 categories, headline text only.

| Model                                          | Accuracy | Macro-F1 | Macro recall |
|------------------------------------------------|----------|----------|--------------|
| TF-IDF + Logistic Regression (baseline)        | 0.72     | 0.63     | 0.59         |
| TF-IDF + Logistic Regression, `class_weight="balanced"` | 0.69 | 0.63 | 0.68         |
| DistilBERT (fine-tuned)                        | –        | –        | –            |

For reference, always predicting the most common class (POLITICS) would score 0.198 accuracy. Full per-class reports for every run are in [results.md](results.md).

---

## Build log

A record of what was done at each step and why.

### Step 1: Explore the data (`explore.py`)

**What:** printed the shape, the column names, the first rows, the count of each category, the null count and the number of duplicate rows.

**Why:** you look at data before you train on it. The category counts reveal class imbalance. Nulls and duplicates have to be fixed before training.

**First attempt: a 50-row toy CSV.** 5 categories × 10 headlines, so perfectly balanced. The full pipeline ran, but test accuracy was **0.20**, which is the same as random guessing with 5 classes. Inspecting the errors showed why: test headlines shared almost no words with the 40 training headlines (vocabulary: 218 words), so the model was matching filler words like *the*, *in* and *with*. **Lesson: TF-IDF can only use words it saw in training, so more data matters more than a fancier model.**

**Switched to the Kaggle dataset** (209,527 rows, 6 columns). Findings:
- **No nulls, but empty strings exist.** `isnull()` doesn't count `""` as missing: 6 headlines are empty and 19,712 `short_description`s are empty.
- **Duplicates depend on what you compare.** Only 13 rows match on *every* column, but many more share the same headline.
- **Heavy imbalance:** POLITICS has 35,602 rows and EDUCATION has 1,014.
- **Messy labels:** renamed categories (`WORLDPOST` / `THE WORLDPOST`), overlapping topics (`WELLNESS` / `HEALTHY LIVING`) and labels that describe the audience rather than the topic (`BLACK VOICES`, `WOMEN`).

### Step 2: Clean the data and split it (`train.py`)

**Label design: 42 categories mapped to 16.** Users can type a headline on any topic, so we kept broad coverage rather than just 5 categories. But we also cleaned up the label set:
- **Merged** renamed or overlapping labels (e.g. `WORLD NEWS` + `WORLDPOST` + `THE WORLDPOST` → `WORLD`, `TECH` + `SCIENCE` → `SCIENCE & TECH`).
- **Dropped** labels that aren't topics: audience labels (`BLACK VOICES`, `QUEER VOICES`, `LATINO VOICES`, `WOMEN`, `FIFTY`), tone labels (`IMPACT`, `GOOD NEWS`, `WEIRD NEWS`) and vague ones (`U.S. NEWS`, `MEDIA`). They aren't put in an "OTHER" category, because such a category would have no consistent vocabulary to learn from.
- **Kept** ARTS separate from ENTERTAINMENT (galleries and books use different words from celebrity news), and kept RELIGION.
- The mapping is written as `new label → [old labels]` (easy to read) and flipped into an `old → new` lookup with a dict comprehension. The original `category` column is kept for debugging.

**A single `text` column.** The model only ever reads `text`. Today it is the headline. Later it becomes headline + description, which means changing one line, with no changes to the pipeline or the app.

**Cleaning order:** map labels → build `text` (with whitespace stripped) → drop empty texts → drop duplicates.
- **Duplicates are defined by `text` only**, because that's all the model sees. Two rows with the same headline but different links are the same "flashcard" to the model. If one lands in train and the other in test, the test score is inflated. A few headlines appeared with *conflicting* labels, a sign that real labels contain human disagreement.
- No manual lowercasing or punctuation removal: `TfidfVectorizer` does this itself, so training and the app process text the same way.
- Result: 180,514 → **179,153 rows** (1,361 removed). LIFESTYLE lost the most (608), mostly recurring column titles.

**Stratified 80/20 split** (`stratify=y`, `random_state=42`): 143,322 train / 35,831 test. Stratifying keeps the same category proportions in both sets, so small classes such as EDUCATION (1.2% of the data, 432 test rows) are fairly represented. A fixed `random_state` makes runs reproducible, so you can tell whether a change actually helped.

### Step 3: Train and evaluate

**Pipeline:** `TfidfVectorizer()` → `LogisticRegression(max_iter=1000)`. Bundling them means the exact same text processing runs during training and prediction, and the vocabulary is learned from the training data only. Vocabulary: **48,691 words**.

**Baseline result:** accuracy 0.72, macro recall 0.59.
- The gap between the two shows the model does well on the big classes and poorly on the small ones. Accuracy weights every *row* equally, while macro averages weight every *class* equally.
- Small classes have **high precision but low recall** (EDUCATION 0.59 / 0.31, RELIGION 0.69 / 0.35, ARTS 0.66 / 0.37). The model rarely predicts them and misses most of them.
- Big classes act as catch-alls (POLITICS recall 0.86, WELLNESS 0.81).

**Experiment log.** Each run of `train.py` appends its classification report to `results.md`, with a timestamp, a short note and the classifier's settings. The file is opened in append mode, so earlier runs are never overwritten.

**Experiment 1: `class_weight="balanced"`.** Mistakes on rare classes cost more during training, in inverse proportion to class size. Only this one setting changed, so the comparison with the baseline is fair.

| | Baseline | Balanced |
|---|---|---|
| Accuracy | 0.72 | 0.69 |
| Macro recall | 0.59 | **0.68** |
| Macro precision | **0.70** | 0.60 |
| Macro-F1 | 0.63 | 0.63 |
| EDUCATION recall | 0.31 | **0.62** |
| RELIGION recall | 0.35 | **0.65** |
| POLITICS recall | **0.86** | 0.70 |

- Small classes roughly **doubled their recall**, at the cost of lower precision. The model now predicts them more often and gets more of them wrong.
- Big classes lost recall but gained precision (POLITICS precision 0.76 → 0.89). They are no longer catch-alls.
- **Macro-F1 didn't change.** The model didn't learn anything new; it moved the trade-off between precision and recall. Better *knowledge* would have to come from better input: more context (e.g. `short_description`), word pairs (bigrams) or a stronger model.

**Decision: keep the balanced model.** Every category should get a fair chance, because the app is meant to handle news on any topic. Giving up 3 points of accuracy is worth it, compared with a model that rarely predicts small topics such as EDUCATION or RELIGION.

**Saving the model.** `joblib.dump(model, "model.joblib")` saves the whole Pipeline (TF-IDF vocabulary + classifier weights) as one 6.9 MB file. The app loads it instead of retraining. Saving both parts together means the classifier can never be paired with the wrong vocabulary. Check: loaded in a fresh Python process, the model predicted `SPORTS` for "Lakers beat Celtics in overtime thriller". The file isn't committed, because `python train.py` can regenerate it. When loading, the scikit-learn version must match the one used for saving, so the version will be pinned before deployment.

### Checkpoint: `.gitignore` and first commit

Committed once Steps 1–3 worked, before starting the app, so there is a known-good state to return to. `.gitignore` excludes `venv/` (520 MB, recreated with `pip install -r requirements.txt`), `data/` (83 MB, downloaded from Kaggle), `model.joblib` (regenerated by `train.py`), `__pycache__/` and `.DS_Store`. Files were added by name, not with `git add .`, and `git status` was checked before committing.

### Step 4: Streamlit app (`app.py`)

**What:** a text box for a headline. The app shows the predicted category, a confidence percentage with a progress bar, a warning below 50% confidence, and the top 3 categories.

**How:**
- The model is loaded once with `@st.cache_resource`. Streamlit re-runs the whole script on every interaction, so without caching the 6.9 MB model would be reloaded on every interaction (each submitted headline, each button click).
- `predict_proba` returns all 16 probabilities. `argmax` picks the winner, and `model.classes_` maps positions to category names. `argsort()[::-1][:3]` gives the top 3.

**What testing showed:**

| Headline | Prediction | Runner-up |
|---|---|---|
| Lakers beat Celtics in overtime thriller | SPORTS 59% | ENTERTAINMENT 4% |
| Apple stock has increased after iPhone launch | SCIENCE & TECH 99% | BUSINESS 0.3% |
| Durga Puja celebration has started in India | WORLD 52% | – |
| Rumor is Angelina Jolie is going to marry again | ENTERTAINMENT 35% ⚠️ | FAMILY 15% |

- **Confident doesn't mean correct.** The Apple headline is really both tech and business, yet the model says 99% tech. Per-word contributions showed why: "apple", "iphone" and "launch" voted strongly for SCIENCE & TECH (≈ +6.8 together), and only "stock" leaned towards BUSINESS (+0.7). Softmax turns that score gap into ≈ 400× odds. **Confidence measures how consistent the training data was, not how ambiguous the headline really is.** The warning can only catch cases where the model itself is torn.
- **Bias from the dataset:** the data is US-centric (HuffPost), so headlines mentioning another country (e.g. "India") lean towards WORLD, regardless of their actual topic.
- **The top 3 list makes uncertainty visible**, e.g. ENTERTAINMENT vs FAMILY for a celebrity wedding.

### Step 6: Log predictions to SQLite (`db.py`)

**Why:** to see what real users type, how the model performs on it, and to collect corrections that can later become training data (a feedback loop).

**Table `predictions`:**

| Column | Purpose |
|---|---|
| `id` | Auto-incrementing key, used to attach feedback to the right row |
| `created_at` | ISO timestamp (sorts correctly as text) |
| `headline`, `predicted`, `confidence` | What was asked and what the model answered |
| `model_version` | Which model answered, so v1 and v2 can be compared on real traffic |
| `is_correct`, `correct_label` | User feedback; empty (NULL) until given |

**Design choices:**
- The database code lives in its own module, so the FastAPI service can reuse it.
- `?` placeholders are used in every query, so user input is stored as data and never run as SQL (prevents SQL injection).
- Streamlit re-runs the script on every interaction, so `st.session_state` remembers the last logged headline. Otherwise each rerun would log the same prediction again.
- `predictions.db` is git-ignored, because it contains what users typed.

**First real log rows** showed typical messy user input: a partly typed headline (`Iphone launc`) was submitted, and still got 98% SCIENCE & TECH from the single word "iphone". User-submitted text and labels need review before they are used for retraining.

**Feedback form.** Below each prediction there is a dropdown preselected with the model's answer, plus a Submit button. If the prediction is right, the user just submits. If it's wrong, they pick the right category first. `is_correct` is inferred as `correct_label == prediction`, and `save_feedback` runs `UPDATE ... WHERE id = ?` on the logged row.
- The dropdown's `key` includes the row id, so it resets for every new headline.
- `st.button` is True only on the rerun triggered by the click, so feedback is saved once per click.
- The feedback code runs *after* logging, so `row_id` always belongs to the current headline.
- Known limitation: a user who submits without reading the dropdown records "correct" even when it isn't.

Verified: "Apple stock has increased after iPhone launch" was stored as `is_correct = 0`, `correct_label = BUSINESS`, and "New Iphone 18 has been launched" as `is_correct = 1`.

---

## Roadmap

- [x] Explore the data
- [x] Clean the text and merge overlapping categories
- [x] Train and evaluate a TF-IDF baseline
- [x] Handle class imbalance and save the model (`model.joblib`)
- [x] `.gitignore` (venv, data, model) and commit
- [x] Streamlit app with a confidence score, a "not sure" warning below 50% and the top 3 categories
- [x] Log predictions and user feedback to SQLite
- [ ] FastAPI `/predict` endpoint
- [ ] Dockerfile, CI/CD (GitHub Actions) and deployment
- [ ] Add `short_description` to the input text
- [ ] React frontend; fine-tune DistilBERT and compare it with the baseline

## License

_TODO: choose a license (e.g. MIT) and add a `LICENSE` file._
