# 26nov task 1 
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import re
import os

from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# PAGE CONFIGURATION

st.set_page_config(
    page_title="VeriFact Analytics",
    layout="wide"
)

st.title("VeriFact Analytics")
st.caption("Fake News Detection | Data Dashboard | NLP Insights")


# LOAD DATA

@st.cache_data
def load_data():
    df = pd.read_csv("FakeNewsNet.csv")

    df = df[['title', 'real']]

    df = df.dropna(subset=['title', 'real'])

    df['title'] = df['title'].astype(str)
    df['real'] = pd.to_numeric(df['real'], errors='coerce')

    df = df.dropna(subset=['real'])
    df['real'] = df['real'].astype(int)

    df = df[df['real'].isin([0, 1])]

    return df


try:
    df = load_data()
except Exception as e:
    st.error(f"Dataset loading failed: {e}")
    st.stop()


# LOAD MODEL
@st.cache_resource
def load_model():
    model = joblib.load("verifact_model.pkl")
    tfidf = joblib.load("verifact_tfidf.pkl")

    return model, tfidf


try:
    model, tfidf = load_model()
except Exception as e:
    model, tfidf = None, None
    st.warning(f"Model files could not be loaded: {e}")


# DATA PREPROCESSING

df['news_type'] = df['real'].map({
    0: 'Fake News',
    1: 'Real News'
})

df['title_length'] = df['title'].apply(len)

df['word_count'] = df['title'].apply(
    lambda x: len(x.split())
)
def get_words(titles):
    all_words = []

    for title in titles:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', str(title).lower())
        words = [word for word in words if word not in ENGLISH_STOP_WORDS]
        all_words.extend(words)

    return Counter(all_words).most_common(10)

# NAVIGATION

page = st.sidebar.selectbox(
    "Choose Analysis",
    [
        "Dashboard",
        "Real vs Fake Comparison",
        "Misinformation Insights",
        "News Prediction"
    ]
)



# TASK 1: DYNAMIC DATA DASHBOARD
if page == "Dashboard":

    st.header("Fake News Detection Dashboard")

    total = len(df)

    fake_count = (df['real'] == 0).sum()
    real_count = (df['real'] == 1).sum()

    fake_percentage = (
        fake_count / total * 100 if total else 0
    )

    real_percentage = (
        real_count / total * 100 if total else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total News", f"{total:,}")
    col2.metric("Fake News", f"{fake_count:,}")
    col3.metric("Real News", f"{real_count:,}")
    col4.metric("Fake News %", f"{fake_percentage:.2f}%")

    st.divider()

    # News distribution

    st.subheader("News Classification Distribution")

    distribution = df['news_type'].value_counts().reset_index()
    distribution.columns = ['News Type', 'Count']

    fig = px.pie(
        distribution,
        names='News Type',
        values='Count',
        hole=0.4,
        title="Fake vs Real News Distribution"
    )

    st.plotly_chart(fig, width="stretch")

    # Headline length

    st.subheader("Headline Length Distribution")

    fig2 = px.histogram(
        df,
        x='word_count',
        color='news_type',
        barmode='overlay',
        title="Word Count Distribution"
    )

    st.plotly_chart(fig2, width="stretch")

    # Classification metrics from independent test data

    st.subheader("Model Performance Metrics")

    st.info(
        "Upload an independent labelled test dataset to calculate "
        "reliable model performance metrics. Do not use the training "
        "dataset as an independent test set."
    )

    test_file = st.file_uploader(
        "Upload test CSV (title, real)",
        type=['csv'],
        key="dashboard_test"
    )

    if test_file:

        test_df = pd.read_csv(test_file)

        if not {'title', 'real'}.issubset(test_df.columns):
            st.error("CSV must contain title and real columns.")

        elif model is None:
            st.error("Model files are unavailable.")

        else:
            test_df = test_df.dropna(subset=['title', 'real'])

            X_test = test_df['title'].astype(str)
            y_test = pd.to_numeric(
                test_df['real'],
                errors='coerce'
            )

            valid = y_test.isin([0, 1])

            X_test = X_test[valid]
            y_test = y_test[valid].astype(int)

            if len(y_test) == 0:
                st.error("No valid test records found.")

            else:
                X_test_vectorized = tfidf.transform(X_test)

                predictions = model.predict(X_test_vectorized)

                accuracy = accuracy_score(
                    y_test, predictions
                )

                precision = precision_score(
                    y_test,
                    predictions,
                    zero_division=0
                )

                recall = recall_score(
                    y_test,
                    predictions,
                    zero_division=0
                )

                f1 = f1_score(
                    y_test,
                    predictions,
                    zero_division=0
                )

                c1, c2, c3, c4 = st.columns(4)

                c1.metric("Accuracy", f"{accuracy*100:.2f}%")
                c2.metric("Precision", f"{precision*100:.2f}%")
                c3.metric("Recall", f"{recall*100:.2f}%")
                c4.metric("F1 Score", f"{f1*100:.2f}%")

                st.text(
                    classification_report(
                        y_test,
                        predictions,
                        zero_division=0
                    )
                )


# TASK 2: REAL VS FAKE COMPARISON

elif page == "Real vs Fake Comparison":

    st.header(" Real vs Fake News Comparison")

    fake_df = df[df['real'] == 0]
    real_df = df[df['real'] == 1]

    # Average length

    st.subheader("1. Average Headline Length")

    comparison = pd.DataFrame({
        "News Type": ["Fake News", "Real News"],
        "Average Words": [
            fake_df['word_count'].mean(),
            real_df['word_count'].mean()
        ]
    })

    fig = px.bar(
        comparison,
        x="News Type",
        y="Average Words",
        color="News Type",
        title="Average Words per Headline"
    )

    st.plotly_chart(fig, width="stretch")

    # Common words

    def get_words(texts):

        words = []

        for text in texts:

            tokens = re.findall(
                r'\b[a-zA-Z]{3,}\b',
                text.lower()
            )

            tokens = [
                word for word in tokens
                if word not in ENGLISH_STOP_WORDS
            ]

            words.extend(tokens)

        return Counter(words).most_common(15)

    st.subheader("2. Frequently Used Words")

    fake_words = get_words(fake_df['title'])
    real_words = get_words(real_df['title'])

    word_col1, word_col2 = st.columns(2)

    with word_col1:

        st.write("Fake News Words")

        fake_words_df = pd.DataFrame(
            fake_words,
            columns=['Word', 'Frequency']
        )

        fig = px.bar(
            fake_words_df,
            x='Frequency',
            y='Word',
            orientation='h',
            title="Common Words in Fake News"
        )

        st.plotly_chart(fig, width="stretch")

    with word_col2:

        st.write("Real News Words")

        real_words_df = pd.DataFrame(
            real_words,
            columns=['Word', 'Frequency']
        )

        fig = px.bar(
            real_words_df,
            x='Frequency',
            y='Word',
            orientation='h',
            title="Common Words in Real News"
        )

        st.plotly_chart(fig, width="stretch")

    # Text comparison

    st.subheader("3. Compare News Headlines")

    fake_sample = st.selectbox(
        "Select Fake News Headline",
        fake_df['title'].head(100).tolist()
    )

    real_sample = st.selectbox(
        "Select Real News Headline",
        real_df['title'].head(100).tolist()
    )

    col1, col2 = st.columns(2)

    with col1:
        st.error(fake_sample)
        st.write("Words:", len(fake_sample.split()))
        st.write("Characters:", len(fake_sample))

    with col2:
        st.success(real_sample)
        st.write("Words:", len(real_sample.split()))
        st.write("Characters:", len(real_sample))


# TASK 3: MISINFORMATION INSIGHTS

elif page == "Misinformation Insights":

    st.header("Misinformation Insights Report")

    fake_df = df[df['real'] == 0]
    real_df = df[df['real'] == 1]

    st.subheader("1. Dataset Overview")

    st.write(f"Total articles analysed: {len(df)}")
    st.write(f"Fake articles: {len(fake_df)}")
    st.write(f"Real articles: {len(real_df)}")

    st.subheader("2. Headline Length Analysis")

    st.write(
        "Average fake headline length:",
        round(fake_df['word_count'].mean(), 2),
        "words"
    )

    st.write(
        "Average real headline length:",
        round(real_df['word_count'].mean(), 2),
        "words"
    )

    st.subheader("3. Common Fake News Terms")

    fake_words = get_words(fake_df['title'])

    for word, count in fake_words[:10]:
        st.write(f"- {word}: {count} occurrences")

    st.subheader("4. Textual Pattern Analysis")

    fake_exclamation = fake_df['title'].str.count('!').mean()
    real_exclamation = real_df['title'].str.count('!').mean()

    fake_uppercase = fake_df['title'].apply(
        lambda x: sum(c.isupper() for c in x)
    ).mean()

    real_uppercase = real_df['title'].apply(
        lambda x: sum(c.isupper() for c in x)
    ).mean()

    st.write(
        "Average exclamation marks in fake headlines:",
        round(fake_exclamation, 3)
    )

    st.write(
        "Average exclamation marks in real headlines:",
        round(real_exclamation, 3)
    )

    st.write(
        "Average uppercase characters in fake headlines:",
        round(fake_uppercase, 2)
    )

    st.write(
        "Average uppercase characters in real headlines:",
        round(real_uppercase, 2)
    )

    st.warning(
        "These are descriptive textual patterns, not proof that "
        "a headline is fake. Textual patterns alone cannot establish "
        "whether a news claim is true."
    )


# NEWS PREDICTION

elif page == "News Prediction":

    st.header(" News Prediction")

    if model is None:

        st.error("Model files are unavailable.")

    else:

        news = st.text_area(
            "Enter a news headline"
        )

        if st.button("Analyse News"):

            if news.strip():

                vectorized = tfidf.transform([news])

                prediction = model.predict(vectorized)[0]

                if int(prediction) == 0:
                    st.error("Prediction: Fake News")

                else:
                    st.success("Prediction: Real News")

            else:
                st.warning("Please enter a headline.")