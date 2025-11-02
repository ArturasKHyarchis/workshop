import sklearn
import pathlib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from joblib import dump
from helpers import calculate_metrics

data_path = pathlib.Path(r'c:\_dev\workshop\data')

train_df = pd.read_csv(data_path.joinpath('train_data.csv'), encoding='utf-8', sep=',')
val_df = pd.read_csv(data_path.joinpath('val_data.csv'),  encoding='utf-8', sep=',')

model = Pipeline([
    ('vectorizer', define vectorizer here),
    ('classifier', define classifier here)
])



# Fit the model
model.fit(train_df['Sentence'], train_df['Sentiment'])

# Evaluate the model
val_labels = val_df['Sentiment']
val_pred = model.predict(val_df['Sentence'])

_, _, _ = calculate_metrics(val_labels, val_pred, print_metrics=True)

# Example prediction for single sentence
print(model.predict(["This is a great product!"]))

# Save the model
model_path = data_path.parent.joinpath('models','sklearn_model.joblib')
dump(model, model_path)
