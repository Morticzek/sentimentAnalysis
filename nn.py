# load Keras model and predict
import time
import nltk
import tensorflow as tf

import pandas as pd
import re

from sklearn.model_selection import train_test_split
from tensorflow import keras as ks
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"

# set the seed for reproducibility
tf.random.set_seed(42)

loaded_model = ks.models.load_model('D:\\JetbrainsProjects\\PyCharmProjects\\sentimentAnalysis\\models\\model_new.h5')
print("Model loaded")
loaded_model.summary()
# loaded_model.load_weights('D:\\JetbrainsProjects\\PyCharmProjects\\sentimentAnalysis\\models\\model_weights.h5')
loaded_model.get_weights()
# initialize a tokenizer and pass the data to be tokenized



tokenizer = Tokenizer()
training_data = 'D:\\JetbrainsProjects\\PyCharmProjects\\sentimentAnalysis\\data\\training.1600000.processed.noemoticon.csv'
df = pd.read_csv(training_data, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)

decode_map = {0: NEGATIVE, 2: NEUTRAL, 4: POSITIVE}


def decoder(label):
    return decode_map[int(label)]


df.target = df.target.apply(lambda x: decoder(x))


df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
tokenizer.fit_on_texts(df_train.text)


def decode_sentiment(score, include_neutral=True):
    if include_neutral:
        label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = NEGATIVE
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE

        return label
    else:
        return NEGATIVE if score < 0.5 else POSITIVE


def predict(text, include_neutral=True):
    start_at = time.time()
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=120)
    # Predict
    score = 1.00 - loaded_model.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(score, include_neutral=include_neutral)

    return {"label": label, "score": float(score), "elapsed_time": time.time()-start_at}


loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


predict("Hello World")
predict("I love this movie")
predict("I hate this movie")