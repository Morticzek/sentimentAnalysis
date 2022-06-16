# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'window.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
import nltk as nltk
from PyQt5 import QtCore, QtGui, QtWidgets

import time
import nltk.stem as st
import tensorflow as tf

import pandas as pd
import re

from sklearn.model_selection import train_test_split
from tensorflow import keras as ks
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# TEXT CLEANING
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

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

    # Return string with sentiment label, score and time
    # return {"label": label, "score": float(score), "elapsed_time": time.time()-start_at}
    return """Input: {text}, Label: {label}, score: {score:.2f}, elapsed time: {elapsed_time:.3f}s\n"""\
        .format(text = text, label = label, score = float(score), elapsed_time = time.time()-start_at)


loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

class Ui_SentimentAnalysis(object):
    def setupUi(self, SentimentAnalysis):
        SentimentAnalysis.setObjectName("SentimentAnalysis")
        SentimentAnalysis.resize(600, 400)
        SentimentAnalysis.setMinimumSize(QtCore.QSize(600, 400))
        SentimentAnalysis.setMaximumSize(QtCore.QSize(600, 400))
        self.centralwidget = QtWidgets.QWidget(SentimentAnalysis)
        self.centralwidget.setObjectName("centralwidget")
        self.runButton = QtWidgets.QPushButton(self.centralwidget)
        self.runButton.setGeometry(QtCore.QRect(80, 110, 131, 31))
        font = QtGui.QFont()
        font.setFamily("Minecraft")
        font.setPointSize(12)
        self.runButton.setFont(font)
        self.runButton.setObjectName("runButton")
        self.clearButton = QtWidgets.QPushButton(self.centralwidget)
        self.clearButton.setGeometry(QtCore.QRect(80, 150, 131, 31))
        font = QtGui.QFont()
        font.setFamily("Minecraft")
        font.setPointSize(12)
        self.clearButton.setFont(font)
        self.clearButton.setObjectName("clearButton")
        self.provideText = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.provideText.setGeometry(QtCore.QRect(30, 40, 221, 51))
        self.provideText.setObjectName("provideText")
        self.resultsText = QtWidgets.QTextBrowser(self.centralwidget)
        self.resultsText.setGeometry(QtCore.QRect(280, 40, 291, 291))
        self.resultsText.setObjectName("resultsText")
        self.resultsLabel = QtWidgets.QLabel(self.centralwidget)
        self.resultsLabel.setGeometry(QtCore.QRect(370, 10, 111, 31))
        font = QtGui.QFont()
        font.setFamily("Minecraft")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.resultsLabel.setFont(font)
        self.resultsLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.resultsLabel.setObjectName("resultsLabel")
        self.provideLabel = QtWidgets.QLabel(self.centralwidget)
        self.provideLabel.setGeometry(QtCore.QRect(40, 10, 201, 21))
        font = QtGui.QFont()
        font.setFamily("Minecraft")
        font.setPointSize(12)
        self.provideLabel.setFont(font)
        self.provideLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.provideLabel.setObjectName("provideLabel")
        SentimentAnalysis.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(SentimentAnalysis)
        self.statusbar.setObjectName("statusbar")
        SentimentAnalysis.setStatusBar(self.statusbar)

        self.retranslateUi(SentimentAnalysis)
        self.clearButton.clicked.connect(self.resultsText.clear) # type: ignore
        self.runButton.clicked.connect(self.pressIt) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(SentimentAnalysis)

    def pressIt(self):
        text = self.provideText.toPlainText()
        result = predict(text)
        self.resultsText.append(result)

    def retranslateUi(self, SentimentAnalysis):
        _translate = QtCore.QCoreApplication.translate
        SentimentAnalysis.setWindowTitle(_translate("SentimentAnalysis", "SentimentAnalysis"))
        self.runButton.setText(_translate("SentimentAnalysis", "Run"))
        self.clearButton.setText(_translate("SentimentAnalysis", "Clear Results"))
        self.resultsLabel.setText(_translate("SentimentAnalysis", "Results:"))
        self.provideLabel.setText(_translate("SentimentAnalysis", "Provide the text here:"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    SentimentAnalysis = QtWidgets.QMainWindow()
    ui = Ui_SentimentAnalysis()
    ui.setupUi(SentimentAnalysis)
    SentimentAnalysis.show()
    sys.exit(app.exec_())
