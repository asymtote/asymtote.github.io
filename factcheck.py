import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


f = pd.read_csv('dataset/Fake.csv')
t = pd.read_csv('dataset/True.csv')

f['class'] = 0
t['class'] = 1

f_mnl_tst = f.iloc[-10:]
t_mnl_tst = t.iloc[-10:]
f = f.iloc[:-10]
t = t.iloc[:-10]

d_mrg = pd.concat([f, t], axis=0)
d = d_mrg.drop(['title', 'subject', 'date'], axis=1)
d = d.sample(frac=1, random_state=42).reset_index(drop=True)

regex_patterns = [
    (re.compile(r'\[.*?\]'), ''),
    (re.compile(r'https?://\S+|www\.\S+'), ''),
    (re.compile(r'<.*?>+'), ''),
    (re.compile(r'\n'), ''),
    (re.compile(r'\w*\d\w*'), ''),
    (re.compile(f"[{re.escape(string.punctuation)}]"), '')
]

def wordopt(text):
    text = text.lower()
    for pattern, replacement in regex_patterns:
        text = pattern.sub(replacement, text)
    return text

d['text'] = d['text'].apply(wordopt)

x = d['text']
y = d['class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=42)

vectorization = TfidfVectorizer(max_features=5000, stop_words='english')
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

lr = LogisticRegression()
lr.fit(xv_train, y_train)
pred_lr = lr.predict(xv_test)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
rf.fit(xv_train, y_train)
pred_rf = rf.predict(xv_test)

def output_label(n):
    return "Fake News" if n == 0 else "True News"

def manual_testing(news,model):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_xv_test = vectorization.transform(new_def_test["text"])
    if model == 'Random Forest':
        pred_rf = rf.predict(new_xv_test)
        rf_accuracy = accuracy_score(y_test,  rf.predict(xv_test))
        return {'res': output_label(pred_rf[0]) ,
                'acc': rf_accuracy*100}
    else:
        pred_lr = lr.predict(new_xv_test)
        lr_accuracy = accuracy_score(y_test,  lr.predict(xv_test))
        return {'res': output_label(pred_lr[0]) ,
                'acc': lr_accuracy*100}