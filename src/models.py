from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

def build_pipeline(model_name="logistic", max_features=5000):
    """
    Factory function to build and return a specific ML pipeline.
    """
    # 1. Define the Feature Extractor (The "Embedding" layer equivalent)
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    
    # 2. Define the Classifier (The "Dense" layers equivalent)
    if model_name == "logistic":
        classifier = LogisticRegression(max_iter=1000, C=1.0)
    
    elif model_name == "svm":
        # Linear kernel is standard for high-dimensional text data
        classifier = LinearSVC(random_state=42, dual=False)
    
    elif model_name == "naive_bayes":
        classifier = MultinomialNB()
    
    else:
        raise ValueError(f"Model '{model_name}' is not supported. Choose from: logistic, svm, naive_bayes")

    # 3. Assemble and return the pipeline
    return Pipeline([
        ('tfidf', tfidf),
        ('classifier', classifier)
    ])