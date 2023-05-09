# CPSC-481-Sentiment-Analyzer-Project
A Simple Sentiment Analyzer to analyze text if either positive, negative, or neutral.

# Created by
* Francisco Bolaños

# Main.py
This is going to be the file of code that evaluates everything. First we must [install nltk](https://www.nltk.org/install.html) in your computer. After you have followed their instructions and downloaded the packages. Usually after you just need a nltk.download(all). That way it will be able to import the packages in the code.
The code will then be able to be processed in the terminal as long as you are in the same directory as the folder and will be able to be run using python Main.py
The code will then present a choice if you want to enter a sentence to evaluate the sentiment or evaluate the csv file included and present the data in a data frame.
You can then exit if youd like by just entering the number 0.

# sm_posts.csv
This is the csv file that contains social media posts from both Twitter and Reddit that mention the college California State University Fullerton in some way. It will contain the date it the message was posted, if it was posted by a human or not, and the message included.