import streamlit as st
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Download NLTK stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Streamlit App
st.title("LDA Topic Modeling with Word Clouds")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(df.head())

    # Ensure 'Text' column exists
    if "Text" in df.columns:
        df = df[["Text"]].dropna()

        # Text preprocessing function
        def preprocess_text(text):
            text = re.sub(r"http\S+|www.\S+", "", text)  # Remove URLs
            text = re.sub(r"[^a-zA-Z ]", "", text)  # Remove special characters
            text = text.lower().strip()  # Convert to lowercase
            text = " ".join(
                [word for word in text.split() if word not in stop_words]
            )  # Remove stopwords
            return text

        df["cleaned_text"] = df["Text"].apply(preprocess_text)

        # Convert text into a document-term matrix
        vectorizer = CountVectorizer(max_features=5000)
        X = vectorizer.fit_transform(df["cleaned_text"])

        # Train LDA model
        num_topics = st.slider(
            "Select number of topics", min_value=2, max_value=10, value=5
        )
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(X)

        # Display topics
        st.subheader("Identified Topics")

        def display_topics(model, feature_names, num_words):
            topics = []
            for topic_idx, topic in enumerate(model.components_):
                topics.append(
                    " ".join(
                        [
                            feature_names[i]
                            for i in topic.argsort()[: -num_words - 1 : -1]
                        ]
                    )
                )
            return topics

        topics = display_topics(lda, vectorizer.get_feature_names_out(), 10)
        for i, topic in enumerate(topics):
            st.write(f"**Topic {i+1}:** {topic}")

        # Word cloud visualization
        st.subheader("Word Clouds for Topics")
        for i, topic in enumerate(lda.components_):
            wordcloud = WordCloud(
                width=800, height=400, background_color="white"
            ).generate(
                " ".join(
                    [
                        vectorizer.get_feature_names_out()[i]
                        for i in topic.argsort()[-20:]
                    ]
                )
            )

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(f"Topic {i + 1}")
            st.pyplot(fig)
    else:
        st.error("CSV file must contain a 'Text' column.")
