from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
import jieba


kw_model = KeyBERT('03_trained_bert')

def chinese_tokenizer(text):
    return list(jieba.cut(text))

def extract_keywords(text):
    vectorizer = CountVectorizer(tokenizer=chinese_tokenizer, ngram_range=(1, 1))
    keywords = kw_model.extract_keywords(text, vectorizer=vectorizer, keyphrase_ngram_range=(1, 1), top_n=30)
    return [keyword[0] for keyword in keywords]

