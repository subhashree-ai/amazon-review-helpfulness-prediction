import numpy as np
import nltk
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

class ColumnExtractor(TransformerMixin):

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xcols = X[self.cols]
        return Xcols
    
    
class TextTFIDFVectorizer(TransformerMixin):
    
    def __init__(self):
        self.vec = None

    def fit(self, X, y=None):
        # stateless transformer
        self.vec =TfidfVectorizer(min_df=0.01, max_df=1., ngram_range=(1, 1))
        self.vec.fit(X['normalize_review_text'])
        return self
    
    def transform(self, X):
        # assumes X is a DataFrame
        # Defining the Tfidf vectorizer
        #vectorizer = TfidfVectorizer(min_df=0.01, max_df=1., ngram_range=(1, 1))
        #fit the vectorizers to the data.
        features = self.vec.transform(X['normalize_review_text'])
    
        return features
        
class TextWord2Vectorizer(TransformerMixin):
    
    def __init__(self):
        self.word2vec = None
        
    def _col_transform(self, words, model, vocabulary, num_features):
        feature_vector = np.zeros((num_features,),dtype="float64")
        nwords = 0.
    
        for word in words:
            if word in vocabulary: 
                nwords = nwords + 1.
                feature_vector = np.add(feature_vector, model[word])

        if nwords:
            feature_vector = np.divide(feature_vector, nwords)

        return feature_vector
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        wpt = nltk.WordPunctTokenizer()
        tokenized_corpus = [wpt.tokenize(document) for document in X['normalize_review_text']] 
        
        # Set values for various parameters
        feature_size = 100    # Word vector dimensionality  
        window_context = 5    # Context window size                                                                                    
        min_word_count = 2    # Minimum word count                        
        sample = 1e-3         # Downsample setting for frequent words

        self.word2vec = Word2Vec(tokenized_corpus, size=feature_size, 
                                      window=window_context, min_count = min_word_count,
                                      sample=sample, iter=10)
        
        vocabulary = set(self.word2vec.wv.index2word)
        features = [self._col_transform(tokenized_sentence, self.word2vec, vocabulary, num_features=100)
                    for tokenized_sentence in tokenized_corpus]
        return np.array(features)
    
    def fit_transform(self, X, y=None):
        return self.transform(X)