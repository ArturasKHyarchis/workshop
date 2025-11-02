import tensorflow as tf
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from helpers import preprocess_sentiment, decode_label

data_path = pathlib.Path(r'c:\_dev\workshop\data')

# Define the model
def create_model(print_summary=True):

    tokenizer_layer = keras.layers.TextVectorization(
    )

    model = keras.models.Sequential(
        [

        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy']
    )
    if print_summary:
        model.summary()
    return model, tokenizer_layer


model, tokenizer_layer = create_model(True)

# Preprocess the data
train_data = pd.read_csv(data_path.joinpath('train_data.csv'), encoding='utf-8')
features = train_data['Sentence'].values

# Adapt the tokenizer layer
tokenizer_layer.adapt(features)

# Encode labels
train_data['labels'] = train_data['Sentiment'].apply(preprocess_sentiment)
labels = train_data['labels'].values
labels_categorical = to_categorical(labels, num_classes=3)



# Fit the model
model.fit(features,
          labels_categorical,
          # validation_data = (val_features, val_labels_categorical),
          epochs=100,
          batch_size=32,
          # callbacks=[early_stopping, reduce_lr]
)

# Evaluate the model
loss, acc = model.evaluate(val_features, val_labels_categorical)
print(f'Test loss: {loss}, Test accuracy: {acc}')

# Example prediction for single sentence
print(model.predict(tf.constant([["This is a great product!"]])))

# Save and load the model
model.export(data_path.parent.joinpath('models','mlp_model.tf'))
loaded_model = tf.saved_model.load(data_path.joinpath('mlp_model.tf'))
predictions = loaded_model.serve(tf.constant([["This is a great product!"]])).numpy()
# prediction = tf.argmax(predictions, axis=1).numpy()[0]
# predicted_label = decode_label([prediction])[0]
# print(f'Predicted label: {predicted_label}')
# print(f'Predicted probabilities: {predictions}')
