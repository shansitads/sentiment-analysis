# :sunny::zap: Problem Statement :zap::sunny:

**NLP Challenge**: IMDB Dataset of 50K Movie Reviews to perform Sentiment analysis

Perform a thorough Exploratory Data Analysis of the dataset and report the final performance metrics for your approach. Suggest ways in which you can improve the model.

View [solution code](./sentiment_analysis.ipynb)

## :pushpin: Exploration + Preprocessing Steps Followed

**Exploration**:

- _Balanced dataset_: Check if number of positive and negative reviews is similar

- _Word frequencies_: Identify frequently appearing words through a word cloud

- _Token frequencies_: Identify frequently appearing tokens to understand cleaning requirements

**Preprocessing & Cleaning**:

- _HTML Tag Removal_: Remove all html </..> tags

- _Lowercasing_: Convert all characters to lowercase to ensure uniformity.

- _Stop Words Removal_: Remove common words that don't contribute much to the sentiment.

- _Stemming/Lemmatization_: Reduce words to their base or root form.

- _Punctuation Removal_: Remove punctuation marks.

- _Handling Special Characters and Numbers_: Decide whether to remove or keep special characters and numbers based on their relevance.

- _Text Normalization_: Expand contractions (e.g., "don't" to "do not").

## :pushpin: Models + Feature Engineering Explored

1. Bidirectional RNN(LSTM) with Embeddings

2. Logistic Regression with TF-IDF Vectorization

## :pushpin: Potential improvements

**Bidirectional RNN with Embeddings**:

- Optimize the `max_len` and `max_words` components during tokenization
- Optimize the `embedding_dim` in Word2Vec embeddings
- Experiment with different Word2Vec embeddings besides the `word2vec-google-news-300`

**Logistic Regression with TF-DIF Vectorization**:

- Optimize the `n_grams` in TF-DIF vectorization

### Feature Engineering - BERT Tokenization

Due to resource limitations, following the pre-trained BERT model for tokenization as a feature engineering methodology was taking too long to run so I decided to continue my project by using a standard Keras tokenizer instead. However, the code is included below if anyone wishes to try it out.

```
import tensorflow as tf

from transformers import BertTokenizer, TFBertModel

# Initialize BERT tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize text data
def bert_tokenize(texts, tokenizer, max_len):
    return tokenizer(texts, padding='max_length', truncation=True, max_length=max_len, return_tensors='tf')

max_len = 100  # Define the maximum sequence length

X_train_bert = bert_tokenize(X_train.tolist(), bert_tokenizer, max_len)
X_test_bert = bert_tokenize(X_test.tolist(), bert_tokenizer, max_len)

# Define the model
input_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
attention_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name='attention_mask')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')
bert_output = bert_model([input_ids, attention_mask])[0]
bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(bert_output)
dropout = tf.keras.layers.Dropout(0.5)(bi_lstm)
bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(dropout)
output = tf.keras.layers.Dense(1, activation='sigmoid')(bi_lstm)

model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit({'input_ids': X_train_bert['input_ids'], 'attention_mask': X_train_bert['attention_mask']},
                    y_train, epochs=3, batch_size=16,
                    validation_data=({'input_ids': X_test_bert['input_ids'], 'attention_mask': X_test_bert['attention_mask']}, y_test))
```
