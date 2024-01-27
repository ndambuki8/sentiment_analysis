from calendar import c
from lib2to3.pgen2 import token
import numpy as np  #Linear algebra

import pandas as pd #data processing, csv fiel i/o  eg. pandas.read_csv

#3Importing the dataset

df = pd.read_csv('tweets_dataset.csv', encoding='ISO-8859-1', header=None)

#Display the top data elements in the csv file
# print(df.head()) 

##Feature Engineering
columns = df.columns
# print(columns) --display what is the headers of the columns

#Removing the unneeded columns
df.drop([1,2,3,4], axis=1, inplace=True) 
# print(df.head()) 

#Renaming the headers of the columns  of the dataframe
df.columns=['sentiment', 'data']
# print(df.head()) 

#pick one of the available data columns in the dataframe - sentiment
y = df['sentiment'] 


##Splitting the dataset into train and test 
from sklearn.model_selection import train_test_split

df_train, df_test, y_train, y_test = train_test_split(df['data'], y, test_size=0.33, random_state=42)

#Confirming the sample sizes selected for training and testing
# print('DF Train Shape: ', df_train.shape)
# print('DF Test Shape: ', df_test.shape)
# print('Y Train Shape: ', y_train.shape)
# print('Y Test Shape: ', y_test.shape)

##Building the deep learning model --Swutch to COLLAB BECAUSE OF RESOURCES 
from keras.preprocessing.text import Tokenizer 
# from keras.preprocessing.text import Tokenizer
max_words = 10000
tokenizer=Tokenizer(max_words)
tokenizer.fit_on_texts(df_train)
sequence_train=tokenizer.texts_to_sequences(df_train)
sequence_test = tokenizer.texts_to_sequences(df_test)

#Vectorization using word2Vec vectorizer
word2vec = tokenizer.word_index
V=len(word2vec)
print('dataset has %s number of independent tokens'  %V)

from keras.preprocessing.sequence import pad_sequences
data_train=pad_sequences(sequence_train)
print(data_train.shape)

T=data_train.shape[1]
data_test=pad_sequences(sequence_test,maxlen=T)
print(data_test.shape)

from keras.layers import Input,Conv1D,MaxPooling1D,Dense,GlobalMaxPooling1D,Embedding
from keras.models import Model

D=20
i=Input((T,))
x=Embedding(V+1,D)(i)
x=Conv1D(32,3,activation='relu')(x)
x=MaxPooling1D(3)(x)
x=Conv1D(64,3,activation='relu')(x)
x=MaxPooling1D(3)(x)
x=Conv1D(128,3,activation='relu')(x)
x=GlobalMaxPooling1D()(x)
x=Dense(5,activation='softmax')(x)
model=Model(i,x)
print(model.summary())

##TRAIBNGING THE MODEL
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
cnn_senti=model.fit(data_train,y_train,validation_data=(data_test,y_test),epochs=1,batch_size=100)


y_pred=model.predict(data_test)
y_pred


y_pred=np.argmax(y_pred,axis=1)
y_pred


from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns

cm=confusion_matrix(y_test,y_pred)
ax=sns.heatmap(cm,annot=True,cmap='Blues',fmt=' ')
ax.set_title('Confusion Matrix')
ax.set_xlabel('y_test')
ax.set_ylabel('y_pred')


print(classification_report(y_test,y_pred))

from sklearn.preprocessing import LabelEncoder
import joblib
# Create and fit the LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(y_train)

# Save the trained LabelEncoder
joblib.dump(label_encoder, 'label_encoder.pkl')


model.save('sentiment_CNN_model.h5')