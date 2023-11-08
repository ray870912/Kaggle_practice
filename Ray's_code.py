#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Dropout
from tensorflow.keras.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives, BinaryAccuracy, Precision, Recall, AUC
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from IPython.display import display
import matplotlib.pyplot as plt

# In[2]:


### Import File ###

# Read txt File With csv and Format in Dataframe
input_train_data = pd.read_csv('Kaggle/kaggle_train_dataset.txt', delimiter = "\t")
input_test_data = pd.read_csv('Kaggle/kaggle_test_dataset.txt', delimiter = "\t")

# Output "Train" Data to Excel
input_train_data.to_excel('Kaggle_training_data.xlsx')
input_test_data.to_excel('Kaggle_test_data.xlsx')


# In[3]:


### Data Pre-Processing ###

# Drop Useless Columns
train_data = input_train_data.drop(columns=['car','toCoupon_GEQ5min'])

# Lebel Encoding
label_encoder = preprocessing.LabelEncoder()

train_data['gender'] = label_encoder.fit_transform(train_data['gender'])
train_data['age'] = label_encoder.fit_transform(train_data['age'])
train_data['income'] = label_encoder.fit_transform(train_data['income'])
train_data['Bar'] = label_encoder.fit_transform(train_data['Bar'])
train_data['CoffeeHouse'] = label_encoder.fit_transform(train_data['CoffeeHouse'])
train_data['CarryAway'] = label_encoder.fit_transform(train_data['CarryAway'])
train_data['RestaurantLessThan20'] = label_encoder.fit_transform(train_data['RestaurantLessThan20'])
train_data['Restaurant20To50'] = label_encoder.fit_transform(train_data['Restaurant20To50'])

# One-Hot Encoding
train_data = pd.get_dummies(train_data, columns=['destination', 'passanger', 'weather', 'temperature',                                                 'time', 'coupon', 'expiration', 'maritalStatus',                                                 'has_children', 'education', 'occupation'])

# Classify X & Y
X = train_data.drop(columns=['Label'])
y = train_data['Label']

# Columns Rename
X.rename(columns={'coupon_Restaurant(<20)': 'coupon_Restaurant(Less20)'}, inplace=True)

# Split Train & Validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=32)


### Prediction ###


# In[4]:


# Setting Evaluating Metrics
METRICS = [
    TruePositives(name='tp'),
    FalsePositives(name='fp'),
    TrueNegatives(name='tn'),
    FalseNegatives(name='fn'), 
    BinaryAccuracy(name='accuracy'),
    Precision(name='precision'),
    Recall(name='recall'),
    AUC(name='auc')
]

"""
# Build the Predict Model

# ANN(accuracy 73.4%)

model = Sequential()
model.add(Dense(512, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=METRICS)

# Fit Data into Model
history = model.fit(X_train, y_train, validation_split=0.2,
                    epochs=50, batch_size=10, shuffle=True)

# Validation Evaluate Model
score = model.evaluate(X_test, y_test, verbose=0)
print("Evaluate on test data")
for name, value in zip(model.metrics_names, score):
    print(name, ': ', value)

plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('Train')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['acc', 'loss'], loc='upper left')
plt.savefig('train.jpg')
plt.show()

plt.plot(history.history['val_accuracy'])
plt.plot(history.history['val_loss'])
plt.title('Test')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['acc', 'loss'], loc='upper left')
plt.savefig('test.jpg')
plt.show()
"""

# In[5]:


# XGBoost(accuracy 74.8%)
xgb_cl = xgb.XGBClassifier()

# Fit
xgb_cl.fit(X_train, y_train)

# Predict
prediction = xgb_cl.predict(X_test)

xgb_cl.score(X_test, y_test)


# In[6]:


"""
# accuracy 68.9%
# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
"""

# In[7]:


input_test_data

input_test_data = input_test_data.drop(columns=['car','toCoupon_GEQ5min'])

# Lebel Encoding
label_encoder = preprocessing.LabelEncoder()

input_test_data['gender'] = label_encoder.fit_transform(input_test_data['gender'])
input_test_data['age'] = label_encoder.fit_transform(input_test_data['age'])
input_test_data['income'] = label_encoder.fit_transform(input_test_data['income'])
input_test_data['Bar'] = label_encoder.fit_transform(input_test_data['Bar'])
input_test_data['CoffeeHouse'] = label_encoder.fit_transform(input_test_data['CoffeeHouse'])
input_test_data['CarryAway'] = label_encoder.fit_transform(input_test_data['CarryAway'])
input_test_data['RestaurantLessThan20'] = label_encoder.fit_transform(input_test_data['RestaurantLessThan20'])
input_test_data['Restaurant20To50'] = label_encoder.fit_transform(input_test_data['Restaurant20To50'])

# One-Hot Encoding
input_test_data = pd.get_dummies(input_test_data, columns=['destination', 'passanger', 'weather', 'temperature',                                                 'time', 'coupon', 'expiration', 'maritalStatus',                                                 'has_children', 'education', 'occupation'])


# In[8]:


# prediction = pd.DataFrame(model.predict_classes(X_test))
prediction = pd.DataFrame(xgb_cl.predict(input_test_data))
prediction.to_excel('predict.xlsx')

