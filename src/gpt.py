import os
import pathlib
import pandas as pd
from openai import AzureOpenAI
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, f1_score
from helpers import calculate_metrics

client = AzureOpenAI(
    azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
    api_key=os.environ.get('AZURE_OPENAI_API_KEY'),
    api_version="2024-05-01-preview",
)

def get_response(prompt):
    try:
        completion = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {
                    "role": "system",
                    "content": "You are an expect at analyzing financial sentiments."
                               "Please evaluate the following sentence for its financial sentiment and respond with one word: 'negative', 'neutral', or 'positive'.",
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f'Error processing prompt: {prompt}. Error: {e["message"]}')
        return 'negative'



data_path = pathlib.Path(r'c:\_dev\workshop\data')

test_data = pd.read_csv(data_path.joinpath('test_data.csv'), encoding='utf-8')
print(f'Start processing at: {datetime.now()}')
test_data['gpt_sentiment'] = test_data['Sentence'].apply(get_response)
print(f'Complete processing at: {datetime.now()}')
test_data.to_csv(data_path.joinpath('gpt_test_data.csv'), index=False)

_, _, _ = calculate_metrics(test_data['Sentiment'], test_data['gpt_sentiment'])
