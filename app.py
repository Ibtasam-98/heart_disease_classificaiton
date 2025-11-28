import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()

        # Tokenize and remove stopwords
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]

        return ' '.join(words)

    def create_model(self, input_dim):
        """Create ANN model"""
        model = Sequential([
            Dense(512, activation='relu', input_shape=(input_dim,)),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=20):
        """Train the model"""
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        return history


def load_sample_data():
    """Load sample Amazon product reviews dataset"""
    # Create sample data (in practice, you would load from a file)
    sample_reviews = [
        "This product is amazing! I love it so much.",
        "Terrible quality, would not recommend to anyone.",
        "Good value for money, works as expected.",
        "Worst purchase ever, complete waste of money.",
        "Excellent product, fast shipping, great quality!",
        "Not what I expected, very disappointed.",
        "Great product, highly recommended!",
        "Poor quality, broke after first use.",
        "Amazing features, worth every penny.",
        "Mediocre product, nothing special.",
        "Outstanding performance, exceeded expectations!",
        "Cheap material, doesn't work properly.",
        "Perfect for my needs, very satisfied.",
        "Horrible customer service and product quality.",
        "Good product but could be better.",
        "Absolutely love it! Best purchase this year.",
        "Waste of money, save your cash.",
        "Reliable and efficient, good purchase.",
        "Not worth the price, very basic.",
        "Fantastic product, would buy again!"
    ]

    sample_sentiments = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]

    return pd.DataFrame({
        'review': sample_reviews,
        'sentiment': sample_sentiments
    })


def main():
    st.title("🤖 ANN-Based Sentiment Analysis System")
    st.markdown("""
    This system uses Artificial Neural Networks to classify customer reviews as **Positive** or **Negative**.
    """)

    # Initialize analyzer
    analyzer = SentimentAnalyzer()

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Mode",
                                    ["Home", "Train Model", "Real-time Analysis", "Model Performance"])

    if app_mode == "Home":
        show_home()
    elif app_mode == "Train Model":
        train_model(analyzer)
    elif app_mode == "Real-time Analysis":
        real_time_analysis(analyzer)
    elif app_mode == "Model Performance":
        show_performance(analyzer)


def show_home():
    st.header("Welcome to Sentiment Analysis System")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### 🎯 Project Overview

        This project implements an **Artificial Neural Network (ANN)** based sentiment analysis system that:

        - 📊 Classifies customer reviews as **Positive** or **Negative**
        - 🔍 Processes and cleans raw text data
        - 🧠 Uses TF-IDF for feature extraction
        - 🤖 Implements ANN for pattern recognition
        - 🌐 Provides real-time analysis through web interface

        ### 🚀 How to Use

        1. **Train Model**: Train the ANN model with sample data
        2. **Real-time Analysis**: Test the model with your own reviews
        3. **Model Performance**: View training metrics and evaluation

        ### 📈 Benefits

        - Helps e-commerce platforms analyze customer feedback
        - Improves product quality and user satisfaction
        - Provides instant sentiment predictions
        """)

    with col2:
        st.image("https://via.placeholder.com/300x400/4B8BBE/FFFFFF?text=Sentiment+Analysis",
                 caption="Sentiment Analysis System")

        st.info("""
        **Quick Start:**
        - Go to **Train Model** to build the ANN
        - Then use **Real-time Analysis** to test reviews
        """)


def train_model(analyzer):
    st.header("🔧 Train Sentiment Analysis Model")

    st.info("Loading sample Amazon product reviews dataset...")

    # Load data
    data = load_sample_data()

    # Display data info
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Reviews", len(data))

    with col2:
        st.metric("Positive Reviews", sum(data['sentiment']))

    with col3:
        st.metric("Negative Reviews", len(data) - sum(data['sentiment']))

    # Show sample data
    if st.checkbox("Show Sample Data"):
        st.dataframe(data)

    # Preprocess data
    st.subheader("Data Preprocessing")

    with st.spinner("Preprocessing text data..."):
        data['cleaned_review'] = data['review'].apply(analyzer.preprocess_text)

        # Split data
        X = data['cleaned_review']
        y = data['sentiment']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Vectorize text
        analyzer.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_tfidf = analyzer.vectorizer.fit_transform(X_train).toarray()
        X_test_tfidf = analyzer.vectorizer.transform(X_test).toarray()

    st.success("✅ Data preprocessing completed!")

    # Model training
    st.subheader("Model Training")

    epochs = st.slider("Number of Epochs", min_value=10, max_value=100, value=20)

    if st.button("🚀 Train ANN Model"):
        with st.spinner("Training Artificial Neural Network..."):
            # Create and train model
            analyzer.model = analyzer.create_model(X_train_tfidf.shape[1])

            # Split training data for validation
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train_tfidf, y_train, test_size=0.2, random_state=42
            )

            # Train model
            history = analyzer.train(X_train_split, y_train_split, X_val_split, y_val_split, epochs)

            # Evaluate model
            train_accuracy = analyzer.model.evaluate(X_train_split, y_train_split, verbose=0)[1]
            test_accuracy = analyzer.model.evaluate(X_test_tfidf, y_test, verbose=0)[1]

            # Store metrics in session state
            st.session_state.model_trained = True
            st.session_state.train_accuracy = train_accuracy
            st.session_state.test_accuracy = test_accuracy
            st.session_state.history = history.history

            # Make predictions
            y_pred = (analyzer.model.predict(X_test_tfidf) > 0.5).astype("int32")
            st.session_state.y_test = y_test
            st.session_state.y_pred = y_pred

        st.success("✅ Model training completed!")

        # Display results
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Training Accuracy", f"{train_accuracy:.2%}")

        with col2:
            st.metric("Test Accuracy", f"{test_accuracy:.2%}")

        # Show training history
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()

        # Loss plot
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()

        st.pyplot(fig)


def real_time_analysis(analyzer):
    st.header("🔍 Real-time Sentiment Analysis")

    if not hasattr(analyzer, 'model') or analyzer.model is None:
        st.warning("⚠️ Please train the model first in the 'Train Model' section!")
        return

    st.markdown("""
    Enter a customer review below to analyze its sentiment. The system will classify it as **Positive** or **Negative**.
    """)

    # Input methods
    input_method = st.radio("Choose input method:",
                            ["Type Review", "Sample Reviews"])

    if input_method == "Type Review":
        user_review = st.text_area("Enter your review:",
                                   placeholder="Type your product review here...",
                                   height=100)

        if st.button("Analyze Sentiment") and user_review:
            analyze_sentiment(analyzer, user_review)

    else:
        sample_reviews = [
            "This product is absolutely fantastic! I love everything about it.",
            "Terrible quality, completely disappointed with this purchase.",
            "It's okay, nothing special but gets the job done.",
            "Amazing value for money, highly recommended!",
            "Worst product ever, save your money and look elsewhere."
        ]

        selected_review = st.selectbox("Choose a sample review:", sample_reviews)

        if st.button("Analyze Selected Review"):
            analyze_sentiment(analyzer, selected_review)


def analyze_sentiment(analyzer, review):
    """Analyze sentiment of a single review"""
    with st.spinner("Analyzing sentiment..."):
        # Preprocess review
        cleaned_review = analyzer.preprocess_text(review)

        # Vectorize
        review_vectorized = analyzer.vectorizer.transform([cleaned_review]).toarray()

        # Predict
        prediction = analyzer.model.predict(review_vectorized)[0][0]
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        # Display results
        st.subheader("Analysis Results")

        col1, col2 = st.columns(2)

        with col1:
            if sentiment == "Positive":
                st.success(f"🎉 Sentiment: {sentiment}")
            else:
                st.error(f"😞 Sentiment: {sentiment}")

            st.metric("Confidence", f"{confidence:.2%}")

        with col2:
            # Confidence bar
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.barh([0], [confidence * 100], color='green' if sentiment == 'Positive' else 'red')
            ax.set_xlim(0, 100)
            ax.set_xlabel('Confidence (%)')
            ax.set_title('Prediction Confidence')
            st.pyplot(fig)

        # Show processed text
        with st.expander("View Processed Text"):
            st.write("**Original Review:**", review)
            st.write("**Cleaned Review:**", cleaned_review)


def show_performance(analyzer):
    st.header("📊 Model Performance")

    if not st.session_state.get('model_trained', False):
        st.warning("⚠️ Please train the model first to see performance metrics!")
        return

    st.subheader("Training Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Training Accuracy", f"{st.session_state.train_accuracy:.2%}")

    with col2:
        st.metric("Test Accuracy", f"{st.session_state.test_accuracy:.2%}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(st.session_state.y_test, st.session_state.y_pred,
                                   target_names=['Negative', 'Positive'])
    st.text(report)

    # Training History Plots
    st.subheader("Training History")

    history = st.session_state.history

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history['accuracy'], label='Training Accuracy')
        ax.plot(history['val_accuracy'], label='Validation Accuracy')
        ax.set_title('Model Accuracy Over Epochs')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history['loss'], label='Training Loss')
        ax.plot(history['val_loss'], label='Validation Loss')
        ax.set_title('Model Loss Over Epochs')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        st.pyplot(fig)


if __name__ == "__main__":
    main()