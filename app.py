import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open('model/model.pkl', 'rb'))

# Page config
st.set_page_config(page_title="Student Rank Predictor", page_icon="🎯")

st.title("🎯 Student Rank Predictor")
st.markdown("Predict your expected rank based on performance")

st.divider()

# Input layout
col1, col2 = st.columns(2)

with col1:
    score = st.slider("Score", 0, 100)
    accuracy = st.slider("Accuracy (%)", 0, 100)

with col2:
    time_taken = st.slider("Time Taken (minutes)", 0, 120)
    attempts = st.slider("Attempts", 1, 5)

st.divider()

# Prediction
if st.button("🚀 Predict Rank"):
    data = np.array([[score, accuracy, time_taken, attempts]])
    prediction = model.predict(data)

    st.success(f"🎉 Predicted Rank: {int(prediction[0])}")

    st.info("Lower rank = Better performance")


if st.checkbox("Show Sample Data Graph"):
    sample_scores = [60, 70, 80, 90, 100]
    sample_ranks = [500, 300, 150, 80, 20]

    fig, ax = plt.subplots()
    ax.plot(sample_scores, sample_ranks)
    ax.set_xlabel("Score")
    ax.set_ylabel("Rank")

    st.pyplot(fig)