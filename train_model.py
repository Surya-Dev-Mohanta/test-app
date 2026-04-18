import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

data = {
    "text": [
        "Win money now!!!",
        "Hello how are you",
        "Free entry in contest",
        "Let's meet tomorrow"
    ],
    "label": [1, 0, 1, 0]
}

df = pd.DataFrame(data)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", MultinomialNB())
])

model.fit(df["text"], df["label"])

joblib.dump(model, "model.pkl")

print("Model created successfully!")
