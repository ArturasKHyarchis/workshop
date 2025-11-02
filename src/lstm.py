import tensorflow as tf
import pathlib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from helpers import preprocess_sentiment

data_path = pathlib.Path(r'c:\_dev\workshop\data')

#define model
def create_model(print_summary=True):

    tokenizer_layer = keras.layers.TextVectorization(

    )

    model = keras.models.Sequential(
        [

        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy']
    )
    if print_summary:
        model.summary()
    return model, tokenizer_layer


model, tokenizer_layer = create_model(True)
train_data = pd.read_csv(data_path.joinpath('train_data.csv'), encoding='utf-8')

# Preprocess the data
features = train_data['Sentence'].values
train_data['labels'] = train_data['Sentiment'].apply(preprocess_sentiment)
labels = train_data['labels'].values

# Encode labels
labels_categorical = to_categorical(labels, num_classes=3)

# Adapt the tokenizer layer
tokenizer_layer.adapt(features)

val_data = pd.read_csv(data_path.joinpath('val_data.csv'), encoding='utf-8')
val_features = val_data['Sentence'].values
val_data['labels'] = val_data['Sentiment'].apply(preprocess_sentiment)
val_labels = val_data['labels'].values
val_labels_categorical = to_categorical(val_labels, num_classes=3)

# Early stopping if accuracy does not improve for 2 consecutive epochs
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    mode='max'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=3)

# Train the model
model.fit(features, labels_categorical, validation_data = (val_features, val_labels_categorical), epochs=100, batch_size=32, callbacks=[early_stopping, reduce_lr])

# Evaluate the model
test_data = pd.read_csv(data_path.joinpath('test_data.csv'), encoding='utf-8')
test_features = test_data['Sentence'].values
test_data['labels'] = test_data['Sentiment'].apply(preprocess_sentiment)
test_labels = test_data['labels'].values
test_labels_categorical = to_categorical(test_labels, num_classes=3)

loss, acc = model.evaluate(test_features, test_labels_categorical)
print(f'Test loss: {loss}, Test accuracy: {acc}')

print(model.predict(tf.constant([["This is a great product!"]])))

model.export(data_path.parent.joinpath('models','lstm_model.tf'))
# loaded_model = tf.saved_model.load(data_path.joinpath('simple_nn_model.tf'))
# predictions = loaded_model.serve(tf.constant([["This is a great product!"]])).numpy()
# prediction = tf.argmax(predictions, axis=1).numpy()[0]
# predicted_label = le.inverse_transform([prediction])[0]
# print(f'Predicted label: {predicted_label}')
# print(f'Predicted probabilities: {predictions}')
