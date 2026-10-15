import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from datetime import datetime
import joblib


df = pd.read_json('data/News_Category_Dataset_v3.json', lines=True)

groups = {
    "POLITICS": ["POLITICS"],
    "ARTS": ["ARTS", "ARTS & CULTURE", "CULTURE & ARTS"],
    "ENTERTAINMENT": ["ENTERTAINMENT", "COMEDY"],
    "WELLNESS": ["WELLNESS", "HEALTHY LIVING"],
    "FAMILY": ["PARENTING", "PARENTS", "WEDDINGS", "DIVORCE"],
    "LIFESTYLE": ["STYLE & BEAUTY", "STYLE", "HOME & LIVING"],
    "TRAVEL": ["TRAVEL"],
    "WORLD": ["WORLD NEWS", "WORLDPOST", "THE WORLDPOST"],
    "FOOD": ["FOOD & DRINK", "TASTE"],
    "BUSINESS": ["BUSINESS", "MONEY"],
    "SPORTS": ["SPORTS"],
    "SCIENCE & TECH": ["TECH", "SCIENCE"],
    "ENVIRONMENT": ["GREEN", "ENVIRONMENT"],
    "CRIME": ["CRIME"],
    "EDUCATION": ["EDUCATION", "COLLEGE"],
    "RELIGION": ["RELIGION"]
}

label_map = {old: new for new, olds in groups.items() for old in olds}

df["label"] = df["category"].map(label_map)
df = df.dropna(subset=["label"])


df['text'] = df['headline'].str.strip()
df = df[df['text'] != '']
df = df.drop_duplicates(subset='text')


print(df.shape)
print(df['label'].value_counts())


X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print('Train:', X_train.shape, 'Test:', X_test.shape)
print(y_test.value_counts())

model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000, class_weight="balanced")),
])

model.fit(X_train, y_train)

print("Vocabulary size:", len(model.named_steps["tfidf"].vocabulary_))
print(model.predict(["Apple launches new iPhone with AI features"]))

y_pred = model.predict(X_test)
note = "class_weight balanced"

report = classification_report(y_test, y_pred)
print(report)

with open("results.md", "a") as f:
    f.write(f"\n## {datetime.now():%Y-%m-%d %H:%M} | {note}\n\n")
    f.write(f"Model: `{model.named_steps['clf']}`\n\n")
    f.write("```\n" + report + "```\n")


# for text, true, pred in zip(X_test, y_test, y_pred):
#     mark = "✓" if true == pred else "✗"
#     print(f"{mark} true={true:14} pred={pred:14} | {text}")

joblib.dump(model, "model.joblib")
print("Saved model.joblib")
