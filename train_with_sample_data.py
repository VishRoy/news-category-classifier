import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv('sample_data.csv')

df['text'] = df['headline'].str.strip()
# print(df[["text", "category"]].head())

X = df['text']
y = df['category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print('Train:', X_train.shape, 'Test:', X_test.shape)
print(y_test.value_counts())

model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000)),
])

model.fit(X_train, y_train)

print("Vocabulary size:", len(model.named_steps["tfidf"].vocabulary_))
print(model.predict(["Apple launches new iPhone with AI features"]))

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

for text, true, pred in zip(X_test, y_test, y_pred):
    mark = "✓" if true == pred else "✗"
    print(f"{mark} true={true:14} pred={pred:14} | {text}")
