#!/usr/bin/env python
# coding: utf-8

# In[]:

import re
import string
import datetime
import contractions
import nltk
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Embedding, GlobalMaxPool1D
from keras.layers.convolutional import Conv1D
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder
from textblob import Word

stopword = set(stopwords.words('english'))
stemmer = nltk.SnowballStemmer("english")

max_words = 500

def clean(text):
    # change to lower case characters
    text = str(text).lower()
    # remove http stuff
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    # remove punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    # remove new line
    text = re.sub('\n', '', text)
    # remove number
    #text = re.sub('\w*\d\w*', '', text)
    text = re.sub('\d', '', text)
    # remove extra space
    text = ' '.join(text.split())
    # remove stop words to focus on important words
    text = ' '.join([word for word in text.split(' ') if word not in stopword])
    # remove number
    text = ' '.join([i for i in text.split(' ') if not i.isdigit()])
    # trim down number of words by snowball stemmer
    #text = ' '.join([stemmer.stem(word) for word in text.split(' ')])
    # lemmatize the text
    text = ' '.join([Word(x).lemmatize() for x in text.split()])
    # change contractions to normal words
    text = contractions.fix(text)
    # correct spelling (too slow)
    #text = ' '.join([Word(x).correct() for x in text.split()])
    return text

# function to create the CNN model
def create_model(num_filters, kernel_size, input_dim, embedding_dim, maxlen, num_neurons):
    model = Sequential()
    model.add(Embedding(input_dim, embedding_dim, input_length=maxlen)) # embedding layer
    model.add(Conv1D(num_filters, kernel_size, padding='same', activation='relu')) # convolutional layer
    model.add(GlobalMaxPool1D()) # max pooling
    model.add(Dense(num_neurons, activation='relu')) # normal NN processing
    model.add(Dense(3, activation='softmax')) # output 3 classification between negative/neutral/positive
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

# function to simulate the accuracy of CNN model with different hyperparameters
def simulate_hyperparam(X_train, y_train, X_test, y_test):
    # Parameter grid for grid search
    param_grid = dict(num_filters=[32, 64, 128],
                      kernel_size=[3, 5, 7],
                      input_dim=[vocab_size],
                      embedding_dim=[50, 100, 150],
                      maxlen=[X_train.shape[1]],
                      num_neurons=[32, 64, 128])
    model = KerasClassifier(build_fn=create_model,
                            epochs=3, batch_size=128,
                            verbose=1)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=2, verbose=1)

    # Evaluate with different hyperparameters
    grid_result = grid.fit(X_train, y_train)
    # Evaluate testing set
    test_accuracy = grid.score(X_test, y_test)

    s = ('Running data set\nBest Accuracy : '
         '{:.4f}\n{}\nTest Accuracy : {:.4f}\n\n')
    output_string = s.format(grid_result.best_score_, grid_result.best_params_, test_accuracy)
    print(output_string)

# load training dataset (60k multi-domain generic sentiment dataset)
# from https://www.kaggle.com/datasets/akgeni/generic-sentiment-multidomain-sentiment-dataset
sentiment_df = pd.read_csv('generic_sentiment_dataset_60k.csv', dtype={'text': 'string'})
sentiment_df = sentiment_df[sentiment_df['text'].notna()]
# clean data
sentiment_df['text'].apply(clean)
# initialize text tokenizer
tokenizer = Tokenizer(split=' ', oov_token='OOV') # assign OOV to out-of-vocabulary words
tokenizer.fit_on_texts(sentiment_df['text'])
# change data from text to numbers
X = tokenizer.texts_to_sequences(sentiment_df['text'])
# pad sequences to fixed length
X = pad_sequences(X, maxlen=max_words)
# total number of vocabulary (+ one pad and OOV)
vocab_size = len(tokenizer.word_index) + 2

# one-hot encoding to encode dataset positive/negative/neutral sentiment
ohe = OneHotEncoder()
y = ohe.fit_transform(sentiment_df[['sentiment']]).toarray()
# ['x0_negative' 'x0_neutral' 'x0_positive']
print(ohe.get_feature_names())
# 80/20 splitting training data and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5)

# simulate different hyperparameters
# simulate_hyperparam(X_train, y_train, X_test, y_test) # uncomment this line to do the hyperparameter simulation

# early stop the training if validation loss decrease (overfitting)
es = EarlyStopping(monitor='val_loss', mode='min', patience=1, verbose=1)

# Model Building
model = create_model(num_filters=64, kernel_size=5,
                     input_dim=vocab_size, embedding_dim=100, maxlen=X_train.shape[1], num_neurons=32)
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

print(model.summary())
# Model Training
model.fit(X_train, y_train, epochs=10, batch_size=128,
          validation_data=(X_test, y_test), callbacks=[es], verbose=1)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# read the assignment busking survey dataset
df = pd.read_csv('Busking survey results.csv', encoding='cp1252')
df = df[df['Busk_Q5_ Provide your comment'].notnull()] # skip null comments survey

comments = df['Busk_Q5_ Provide your comment']
# clean the data
comments = comments.apply(clean)
# change data from text to sequences
pad_comments = tokenizer.texts_to_sequences(comments)
# pad sequences
pad_comments = pad_sequences(pad_comments, maxlen=max_words)
# do the prediction
predictions = model.predict(pad_comments)
# convert one hot encoding back to text categories
df['sentiment'] = ohe.inverse_transform(predictions)

# write the result to csv file
output_csv = 'CNN_result_{}.csv'.format(datetime.datetime.now().strftime("%Y-%m-%d %H%M%S"))
df.to_csv(output_csv)
print(f"Prediction output to {output_csv}")