import streamlit as st
from joblib import load
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords

# Load your logistic regression model and tdidfVectorizer 
lr_loaded = load('logistic_regression_model.joblib')
tv_loaded = load('tfidfVectorizer.joblib')

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Create a function to clean text
def processWord(script):
    script = script.lower()  # Lower case 
    script = re.sub('\[.*?\]', '', script)  # Remove anything with and within brackets
    script = re.sub('\\W', ' ', script)  # Removes any character not a letter, digit, or underscore
    script = re.sub('https?://\S+|www\.\S+', '', script)  # Removes any links starting with https
    script = re.sub('<.*?>+', '', script)  # Removes anything with and within < >
    script = re.sub('[%s]' % re.escape(string.punctuation), '', script)  # Removes any string with % in it 
    script = re.sub('\n', '', script)  # Remove new lines
    script = re.sub('\w*\d\w*', '', script)  # Removes any string that contains at least a digit with zero or more characters
    # Remove stopwords (split the script - > filter out stopwords -> join the words)
    script = ' '.join([word for word in script.split() if word not in stop_words])  
    return script

# Prediction function 
def news_prediction(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test['text'] = new_def_test['text'].apply(processWord)
    new_x_test = new_def_test['text']
    new_tfidf_test = tv_loaded.transform(new_x_test)
    pred_lr = lr_loaded.predict(new_tfidf_test)
    
    if pred_lr[0] == 0:
        return "This is Fake News! Don't Listen what the kopitiam uncle and aunty say."
    else:
        return "The News seems to be True!"

# Streamlit application starts here 
def main():
    # Title of your web app
    st.title("Fake News Prediction System")
    user_text = st.text_area("Enter a sentence to check if it's true or fake:", height=350)
   
    if st.button("Predict"):
        if user_text.strip():  # Check if the input text is not just empty or spaces
            news_pred = news_prediction(user_text)
            if news_pred.startswith("This is Fake News!"):  # Adjusted condition here
                st.error(news_pred, icon="🚨")
            else:
                st.success(news_pred)
                st.balloons()
        else:
            st.error("Please enter some text before analyzing the news article!")  # Updated error message for clarity

if __name__ == "__main__":
    main()
