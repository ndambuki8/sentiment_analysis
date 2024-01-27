import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  
import seaborn as sns 
import nltk 
nltk.download('punkt')

from nltk.tokenize import word_tokenize 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential 
from keras.layers import Embedding, LSTM, Dense, Dropout 
from sklearn.preprocessing import LabelEncoder 

import warnings
warnings.filterwarnings('ignore')
sns.set()

# Load the dataset
imdb = pd.read_csv('tweets_dataset.csv', encoding='ISO-8859-1', header=None)

# Drop columns 2, 3, and 4
imdb = imdb.drop(columns=[1, 2, 3, 4])

# Rename columns 0 and 5
imdb.rename(columns={0: 'sentiment', 5: 'data'}, inplace=True)

# Preprocessing
X = imdb['data']
y = imdb['sentiment']


corpus = [] 
for text in X:
          words = [word.lower() for word in word_tokenize(text)]
          corpus.append(words)

num_words = len(corpus)
# print(num_words)

# imdb.shape

##Split data to train 80% and test 20%

train_size = int(imdb.shape[0] * 0.8)
X_train = imdb.data[:train_size]
Y_train = imdb.sentiment[:train_size]

X_test = imdb.data[train_size:]
Y_test = imdb.sentiment[train_size:]

##Tokenizing the words and padding for equal input dimensions
tokenizer = Tokenizer(num_words)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(X_train, maxlen=128, truncating='post', padding='post')

# print(X_train[0], len(X_train[0]))

X_test = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, maxlen=128, truncating='post', padding='post')

# print(X_test[0], len(X_test[0]))

# print(X_train.shape, Y_train.shape)
# print(X_test.shape, Y_test.shape)

#Label encoding for the Y values
le = LabelEncoder()
Y_train = le.fit_transform(Y_train)
Y_test = le.transform(Y_test)

#Creating a base model
model = Sequential()

model.add(Embedding(input_dim=num_words, output_dim=100, input_length=128, trainable=True))
model.add(LSTM(100, dropout=0.1, return_sequences=True))
model.add(LSTM(100, dropout=0.1))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

history = model.fit(X_train, Y_train, epochs=2, batch_size=64, validation_data=(X_test, Y_test))

validation_sentence = ['This movie was not good at all. It had some good parts like acting was pretty good but story was not impressive at all.']
validation_sentence_tokenized = tokenizer.texts_to_sequences(validation_sentence)
validation_sentence_padded = pad_sequences(validation_sentence_tokenized, maxlen=128, truncating='post', padding='post')
print(validation_sentence[0])
print("Probability of positive sentiment: {}".format(model.predict(validation_sentence_padded)[0]))


model.save('sentiment_TWEETER_LSTM_model.h5')