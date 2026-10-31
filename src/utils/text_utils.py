import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    text = text.lower()
    words = text.split()
    #stopwords removal
    words = [w for w in words if w not in stop_words]
    lemmatized = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(lemmatized)