from lxml import etree
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
import nltk
from collections import defaultdict
import string
import re
import pandas as pd

tree = etree.parse('data/news.xml')
root = tree.getroot()
corpus = root[0]

wnl = WordNetLemmatizer()
data = pd.DataFrame({'Title':[], 'Text':[]})

for news in corpus:
    title = news[0].text
    text = news[1].text

    text_tokenized = word_tokenize(text.lower())
    text_lemmatized = [wnl.lemmatize(word) for word in text_tokenized]
    text_lemmatized = [wnl.lemmatize(word) for word in text_lemmatized if word not in stopwords.words('english')]
    punct = f'^[{re.escape(string.punctuation)}]+$'
    no_punctuation = [re.sub(punct, '', word) for word in text_lemmatized]
    nouns = [word for word in no_punctuation if nltk.pos_tag([word])[0][1] == 'NN']
    data = data.append({'Title': title, 'Text': ' '.join(nouns)}, ignore_index=True)

    # words_counter = defaultdict(int)
    # for word in nouns:
    #     words_counter[word] += 1
    #
    # top5 = sorted(words_counter.items(), key=lambda x: (x[1], x[0]), reverse=True)[1:6]
    # print(title + ':')
    # for word, count in top5:
    #     print(word, end=' ')
    #
    # print('\n')


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(input='content')

tfidf_matrix = vectorizer.fit_transform(data['Text'])
tf_Idf = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names())
for i in range(10):
    print(data.Title[i] + ':')
    print(' '.join(tf_Idf.loc[i, :].nlargest(5).index))
    print()

