import nltk
from typing import List
from pprint import pprint

stopwords = set(nltk.corpus.stopwords.words("english"))
text = nltk.Text(nltk.corpus.state_union.words())
concordance_list = text.concordance_list("america", lines = 2)
#for entry in concordance_list:
 #   print(entry.line)
words: List[str] = nltk.word_tokenize(
     """Beautiful is better than ugly.
     Explicit is better than implicit.
    Simple is better than complex.""" )
text = nltk.Text(words)
fd = text.vocab()
fd.tabulate(3)