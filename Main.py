import pandas as pd  
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

stopwords = nltk.corpus.stopwords.words("english")
regToken = RegexpTokenizer('\w+')
wnl = WordNetLemmatizer() 
sid = SentimentIntensityAnalyzer()
def polarity_scores(text):
    scores = sid.polarity_scores(text)
    compounded_score = scores['compound']      
    return compounded_score


def social_media_csv(): 
    df = pd.read_csv("sm_posts.csv")
    pd.options.display.max_colwidth = 100
    df['Text'] = df['Text'].apply(regToken.tokenize)
    df['Text'] = df['Text'].apply(lambda x: [item for item in x if item not in stopwords])
    df['Text_String'] = df['Text'].apply(lambda x: ' '.join([item for item in x if len(item)>1]))
    df['Text_Lemmatized'] = df['Text_String'].apply(wnl.lemmatize)
    df['Sentiment_Score'] = df['Text_Lemmatized'].apply(polarity_scores) 
    df['Pos/Neg/Neu'] = df['Sentiment_Score'].apply(lambda x: 'positive' if x > .45 else 'neutral' if x >= 0 else 'negative')
    print(df)
    df = df.drop(['Date', 'Human','Text'], axis = 'columns')
    print(df)
    
    
if __name__ == "__main__":
    print("Welcome to the Sentiment Analyzer\n")
    print("Select a choice\n")
    print("1. Enter sentence to get it rated")
    print("2. Check a saved csv file filled with posts about CSUF.")
    print("0. Exit the application\n")
    choice = int(input("Choice: "))
    while True:
        if choice == 1:
            rate_sentence = input("Enter a sentence: ")
            score = polarity_scores(rate_sentence)
            print(score)
            if score > .45:
                print("Positive")
            elif score >= 0:
                print("Neutral")
            else:
                print("Negative") 
            choice = int(input("Choice: "))
        elif choice == 2:
            social_media_csv()
            choice = int(input("Choice: "))
        elif choice == 0:
            print("Thank you for using the analyzer, goodbye!")
            break
        else:
            print("Not an available choice please try again")
            choice = int(input("Choice: "))