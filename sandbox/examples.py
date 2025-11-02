### code examples
### DO NOT RUN THIS FILE!!!

#############################################################################################################
test_data = df.sample(frac=0.2)
data = df.drop(test_data.index)
val_data = data.sample(frac=0.25)
train_data = data.drop(val_data.index)

test_data.to_csv(test_data_path, index=False, encoding='utf-8', sep=',', quotechar='"')
val_data.to_csv(val_data_path, index=False, encoding='utf-8', sep=',', quotechar='"')
train_data.to_csv(train_data_path, index=False, encoding='utf-8', sep=',', quotechar='"')

#############################################################################################################
model = Pipeline([
    ('vectorizer', TfidfVectorizer(max_features=1000)),
    ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
])

model = Pipeline([
    ('vectorizer', CountVectorizer(max_features=1000)),
    ('classifier', RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1))
])

model = Pipeline([
    ('vectorizer', CountVectorizer(max_features=1000)),
    ('classifier', LogisticRegression(max_iter=200, n_jobs=-1))
])

model = Pipeline([
    ('vectorizer', TfidfVectorizer(max_features=1000)),
    ('classifier', SVC())
])

#############################################################################################################
tokenizer_layer = keras.layers.TextVectorization(
    max_tokens=2000,
    pad_to_max_tokens=True,
    output_mode='tf_idf',
    encoding='utf-8',
    standardize='lower_and_strip_punctuation',
)

model = keras.models.Sequential(
    [
        keras.layers.Input(shape=(1,), dtype=tf.string),
        tokenizer_layer,
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(3, activation='softmax'),
    ]
)

#############################################################################################################
# Preprocess validation data
val_data = pd.read_csv(data_path.joinpath('val_data.csv'), encoding='utf-8')
val_features = val_data['Sentence'].values
val_data['labels'] = val_data['Sentiment'].apply(preprocess_sentiment)
val_labels = val_data['labels'].values
val_labels_categorical = to_categorical(val_labels, num_classes=3)

# Early stopping if accuracy does not improve for 5 consecutive epochs and reduce learning rate on plateau
early_stopping = EarlyStopping(
    monitor='accuracy',
    patience=5,
    restore_best_weights=True,
    mode='max'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=3)

#############################################################################################################
tokenizer_layer = keras.layers.TextVectorization(
        max_tokens=2000,
        pad_to_max_tokens=True,
        output_mode='int',
        encoding='utf-8',
        standardize='lower_and_strip_punctuation',
    )

    model = keras.models.Sequential(
        [
            keras.layers.Input(shape=(1,), dtype=tf.string),
            tokenizer_layer,
            keras.layers.Embedding(input_dim=2000, output_dim=512),
            keras.layers.LSTM(512, return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(128, return_sequences=False),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(3, activation='softmax'),
        ]
    )

#############################################################################################################
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

#############################################################################################################

