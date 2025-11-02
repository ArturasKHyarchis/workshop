import pathlib
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, f1_score
from joblib import load
from datetime import datetime
from helpers import decode_label, calculate_metrics

data_path = pathlib.Path(r'c:\_dev\workshop\data')


def get_single_prediction_from_sklean(model, text):
    pred = model.predict([text])
    return pred[0]

def get_single_prediction_from_tf(model, text):
    model_output = model.serve(tf.constant([[text]])).numpy()
    prediction = tf.argmax(model_output, axis=1).numpy()[0]
    return decode_label(prediction)


df = pd.read_csv(data_path.joinpath('test_data.csv'), encoding='utf-8')

# get sklearn model predictions
sklearn_model = load(data_path.parent.joinpath('models','sklearn_model.joblib'))
print(f'Start sklearn processing at: {datetime.now()}')
df['sklearn_sentiment'] = df['Sentence'].apply(lambda x: get_single_prediction_from_sklean(sklearn_model, x))
print(f'Complete sklearn processing at: {datetime.now()}')


# get tensorflow (mlp) model predictions
mlp_model = tf.saved_model.load(data_path.parent.joinpath('models','mlp_model.tf'))
print(f'Start mlp processing at: {datetime.now()}')
df['mlp_sentiment'] = df['Sentence'].apply(lambda x: get_single_prediction_from_tf(mlp_model, x))
print(f'Complete mlp processing at: {datetime.now()}')

# get tensorflow (lstm) model predictions
lstm_model = tf.saved_model.load(data_path.parent.joinpath('models','lstm_model.tf'))
print(f'Start lstm processing at: {datetime.now()}')
df['lstm_sentiment'] = df['Sentence'].apply(lambda x: get_single_prediction_from_tf(lstm_model, x))
print(f'Complete lstm processing at: {datetime.now()}')

df.to_csv(data_path.joinpath('model_compare.csv'), index=False)

# calculate metrics for sklearn model
print('Sklearn model metrics:')
_, _, _ = calculate_metrics(df['Sentiment'], df['sklearn_sentiment'], print_metrics=True)

# calculate metrics for tensorflow (mlp) model
print('TensorFlow model (mlp) metrics:')
_, _, _ = calculate_metrics(df['Sentiment'], df['mlp_sentiment'], print_metrics=True)

# calculate metrics for tensorflow (lstm) model
print('TensorFlow model (lstm) metrics:')
_, _, _ = calculate_metrics(df['Sentiment'], df['lstm_sentiment'], print_metrics=True)


