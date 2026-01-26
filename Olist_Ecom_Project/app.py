import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from deep_translator import GoogleTranslator
from src.preprocess import clean_text

# Page Config.
st.set_page_config(page_title="Olist AI Dashboard", layout="wide")

# Load Models
@st.cache_resource
def load_models():
    base_path = 'models'
    
    # Load NLP Files
    nlp_model = joblib.load(os.path.join(base_path, 'sentiment_model.pkl'))
    vectorizer = joblib.load(os.path.join(base_path, 'tfidf_vectorizer.pkl'))
    
    # Load Clustering File
    kmeans_model = joblib.load(os.path.join(base_path, 'customer_segmentation.pkl'))
    scaler_model = joblib.load(os.path.join(base_path, 'rfm_scaler.pkl'))
    
    return nlp_model, vectorizer, kmeans_model, scaler_model

# Load Models
try:
    nlp_model, tfidf, kmeans, scaler = load_models()
except FileNotFoundError:
    st.error("Models not found! Please check the 'models/' path.")
    st.stop()
    

# Sidebar
st.sidebar.title("Olist E-commerce")
st.sidebar.info("This app uses AI to analyze Customer Sentiment and Segment Customers based on behavior.")
app_mode = st.sidebar.selectbox("Choose Module", ["Home", "Review Sentiment Analyzer", "Customer Segmentation"])

# PAGE 1: HOME
if app_mode == 'Home':
    st.title("Olist Supply Chain AI")
    st.markdown("""
    Welcome to the **Olist Analytics Dashboard**. This project utilizes Machine Learning to optimize supply chain and customer experience.
    
    ### Available Modules:
    1. **Sentiment Analysis (NLP):** Detects if a customer review is Positive or Negative.
    2. **Customer Segmentation (Clustering):** Groups customers into VIP, Lost, or Active categories.
    
    **Techniques Used:** TF-IDF, Logistic Regression, K-Means Clustering, RFM Analysis.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", width=100)
    
# PAGE 2: NLP SENTIMENT
elif app_mode == "Review Sentiment Analyzer":
    st.header("NLP: Customer Review Sentiment")
    st.write("Enter a customer review (Portuguese or English) to see if it's Positive or Negative.")
    
    user_text = st.text_area("Customer Review:", "O produto chegou quebrado e com atraso.")
    
    if st.button("Analyze Sentiment"):
        if user_text:
            try:
                # 1. TRANSLATION LAYER (The Magic ✨)
                # User ka text -> Portuguese ('pt') mein convert karo
                translated_text = GoogleTranslator(source='auto', target='pt').translate(user_text)
                
                st.info(f"🔤 Translated to Portuguese for AI: '{translated_text}'")

                # 2. Clean (Ab Portuguese text ko clean karenge)
                cleaned_text = clean_text(translated_text)
                
                # 3. Vectorize & Predict
                vec_text = tfidf.transform([cleaned_text])
                pred = nlp_model.predict(vec_text)
                prob = nlp_model.predict_proba(vec_text)
                
                # Result Display
                sentiment = "Positive 😊" if pred[0] == 1 else "Negative 😡"
                confidence = np.max(prob) * 100
                
                st.subheader(f"Prediction: {sentiment}")
                st.write(f"**Confidence:** {confidence:.2f}%")
                
                if pred[0] == 0:
                    st.error("Alert: Unhappy Customer!")
                else:
                    st.success("Happy Customer!")

            except Exception as e:
                st.error(f"Translation Error: {e}. Please check internet connection.")
        else:
            st.warning("Please enter some text.")
            
# PAGE 3: CLUSTERING
elif app_mode == "Customer Segmentation":
    st.header("AI Customer Grouping (RFM)")
    st.write("Enter Customer Behavior data to find their segment.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        recency = st.number_input("Recency (Days since last order)", min_value=0, value=30)
    with col2:
        frequency = st.number_input("Frequency (Total Orders)", min_value=1, value=1)
    with col3:
        monetary = st.number_input("Monetary (Total Spent $)", min_value=0.0, value=100.0)
        
    if st.button("Identify Segment"):
        # 1. Prepare Data & Scale
        user_data = np.array([[recency, frequency, monetary]])
        scaled_data = scaler.transform(user_data)
        
        # 2. Predict Cluster
        cluster_id = kmeans.predict(scaled_data)[0]
        
        # 3. Map to Name
        cluster_names = {
            0: 'Active Customer',
            1: 'Lost/Dormant Customer',
            2: 'Loyalist (Regular)',
            3: 'Big Spender (Whale)'
        }
        
        segment_name = cluster_names.get(cluster_id, "Unknown")
        
        st.subheader(f"Customer Belongs to: **{segment_name}**")
        
        if cluster_id == 1:
            st.warning("Strategy: Send 'We Miss You' Email with Coupon.")
        elif cluster_id == 3:
            st.success("Strategy: Offer Premium Support & Loyalty Points.")
        else:
            st.info("Strategy: Standard Engagement.")