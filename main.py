import streamlit as st
import pickle

# Load trained model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📊 Real-Time NLP Sentiment Analysis")

# User Input
user_text = st.text_area("Enter your text here:")

if st.button("Analyze Sentiment"):
    if user_text.strip() != "":
        # Transform text
        text_vec = vectorizer.transform([user_text])
        prediction = model.predict(text_vec)[0]

        if prediction == "positive":
            st.success("😊 Positive Sentiment")
        elif prediction == "negative":
            st.error("😡 Negative Sentiment")
        else:
            st.info("😐 Neutral Sentiment")
    else:
        st.warning("Please enter some text.")
