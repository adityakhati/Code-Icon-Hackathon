from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
import pickle
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB


df = pd.read_csv('train.csv')

lambda x: x*10 if x<2 else (x**2 if x<4 else x+10)
df['label'] = df['category'].apply(lambda x: 0 if x=='beverage' else (1 if x=='condiment' else (2 if x=='appetizer' else (3 if x=='dessert' else 4))))


X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size = 0.2)

#print(X_test)  

cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)


word_freq_df = pd.DataFrame(X_train_cv.toarray(), columns=cv.get_feature_names())
top_words_df = pd.DataFrame(word_freq_df.sum()).sort_values(0, ascending=False)

naive_bayes = MultinomialNB()
model=naive_bayes.fit(X_train_cv, y_train)
predictions = naive_bayes.predict(X_test_cv)


print('Accuracy score: ', accuracy_score(y_test, predictions))
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
pickle.dump(cv,open('counterv.sav','wb'))

df1 = pd.read_csv('validation.csv')

X_test=df1['text']

X_test_cv = cv.transform(X_test)
predictions = naive_bayes.predict(X_test_cv)

testing_predictions = []
for i in range(len(X_test)):
    if predictions[i] == 0:
        testing_predictions.append('beverage')
    elif predictions[i] == 1:
        testing_predictions.append('condiment')
    elif predictions[i] == 2:
        testing_predictions.append('appetizer')
    elif predictions[i] == 3:
        testing_predictions.append('dessert')
    elif predictions[i] == 4:
        testing_predictions.append('snacks')
    
check_df = pd.DataFrame({'Id':df1['Id'],'prediction': testing_predictions})
check_df.replace(to_replace=0, value='beverage', inplace=True)
check_df.replace(to_replace=1, value='condiment', inplace=True)
check_df.replace(to_replace=2, value='appetizer', inplace=True)
check_df.replace(to_replace=3, value='dessert', inplace=True)
check_df.replace(to_replace=4, value='snacks', inplace=True)

filename = "new1.csv"
  
# # writing to csv file 
# with open(filename, 'w') as csvfile: 
#     # creating a csv writer object 
#     csvwriter = csv.writer(csvfile) 
      
#     # writing the fields 
#     #csvwriter.writerow(fields) 
      
#     # writing the data rows 
#     csvwriter.writerow(check_df)

check_df.to_csv('new1.csv')
