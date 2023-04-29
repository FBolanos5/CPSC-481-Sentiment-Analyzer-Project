import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


sentences = ["This analyzer is awesome!", "This anaylzer sucks!", "This analyzer is okay"]
pos_counter = 0
neg_counter = 0
neu_counter = 0
for sentence in sentences:
    sid = SentimentIntensityAnalyzer()
    print(sentence)
    ss = sid.polarity_scores(sentence)
    if ss['compound'] > .5:
        pos_counter += 1 
    elif ss['compound'] > 0:
        neu_counter += 1
    else:
        neg_counter += 1
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end='')    
    print()
print("Amount Positive: ", pos_counter)
print("Amount Neutral: ", neu_counter)
print("Amount Negative: ", neg_counter)