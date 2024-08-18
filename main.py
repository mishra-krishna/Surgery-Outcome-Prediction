import streamlit as st
import pandas as pd
import numpy as np
import joblib
from gensim.models import KeyedVectors
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import string

# Load models
try:
    bilstm_model = load_model('bilstm_model.h5')
    gradient_boosting_model = joblib.load('gradient_boosting_model.pkl')
    word2vec_model = KeyedVectors.load_word2vec_format('word2vec_model.bin', binary=True)
    st.write("Models loaded successfully.")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Create word-to-index mapping
word_index = {word: i + 1 for i, word in enumerate(word2vec_model.index_to_key)}

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def text_to_sequence(tokens, word_index):
    return [word_index.get(token, 0) for token in tokens]

def predict_surgery_type(doc_note):
    preprocessed_text = preprocess_text(doc_note)
    tokens = preprocessed_text.split()
    sequences = text_to_sequence(tokens, word_index)
    max_sequence_length = 100  # Adjust based on your model's requirement
    X = pad_sequences([sequences], maxlen=max_sequence_length, padding='post', truncating='post')
    y_pred = bilstm_model.predict(X)
    y_pred_binary = (y_pred > 0.5).astype(int)
    surgery_types = ['Surgery Type_DALK', 'Surgery Type_EK', 'Surgery Type_PK', 'Surgery Type_THPK']
    return dict(zip(surgery_types, y_pred_binary[0]))

def predict_outcome(features):
    df = pd.DataFrame([features])
    outcome_pred = gradient_boosting_model.predict(df)
    return 'Pass' if outcome_pred[0] == 1 else 'Fail'

# Streamlit App
st.title('Surgery Prediction Dashboard')

st.header('Predict Surgery Outcome from Doctor Notes')
doc_note = st.text_area("Enter Doctor's Notes:")
age = st.number_input('Age', min_value=0, step=1)
gender = st.selectbox('Gender', ['Male', 'Female'])
days_since_last_visit = st.number_input('Days Since Last Visit', min_value=0, step=1)

if st.button('Predict Surgery Outcome'):
    if doc_note and age is not None and gender and days_since_last_visit >= 0:
        try:
            # Predict surgery types
            surgery_predictions = predict_surgery_type(doc_note)
            
            # Prepare features for gradient boosting model
            gender_num = 1 if gender == 'Male' else 0
            features = {
                'Age': age,
                'Gender': gender_num,
                'Days since last visit': days_since_last_visit,
                'Surgery Type_DALK': surgery_predictions['Surgery Type_DALK'],
                'Surgery Type_EK': surgery_predictions['Surgery Type_EK'],
                'Surgery Type_PK': surgery_predictions['Surgery Type_PK'],
                'Surgery Type_THPK': surgery_predictions['Surgery Type_THPK']
            }
            
            # Predict outcome
            outcome = predict_outcome(features)
            st.write(f"Predicted Surgery Outcome: {outcome}")
        except Exception as e:
            st.error(f"Error predicting surgery outcome: {e}")
    else:
        st.error("Please enter all the required details.")
