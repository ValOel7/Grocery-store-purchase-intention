pip install streamlit
import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.title("Grocery store purchase intention predictor")
st.write(
    "Please answer the following questions by choosing the most suitable option, after answering all the questions, press the predict button to form a prediction on the purchase intention"
)
# Load the trained Random Forest model from the pickle file
model = pd.read_pickle('rf.pkl')
# Example questions to collect input
questions = {
    "PS1": "If prices increase, how likely are you to buy from the store?",
    "PS3": "If another grocery store offers cheaper prices, how likely are you to buy from the competitor?",
    "PE2": "How would you describe the cleanliness of the shopping environment in this store?",
    "PPQ1": "How would you rate the overall quality of the products in the store?",
    "PPQ2": "How would you rate the quality of the fresh produce in the store?",
    "PPQ3": "How would you rate the quality of the meat department in the store?",
    "CT2": "How well does the store meet your needs?",
    "CT5": "How consistent is the store in providing good quality products?",
    "PV1": "How would you rate the value for money of the products in the store?",
    "PV2": "How affordable are the products in the store?",
    "PV3": "How much do you save money by shopping at this store compared to others?",
    "PI1": "How likely are you to purchase from this store again?",
    "PI2": "How likely are you to repeat your shopping experience at this store?",
    "PI3": "How likely are you to purchase from this store in the future?",
    "PI4": "How likely are you to recommend this store to others?"
}
# Define the function for predicting purchase intention
def predict_purchase_intention(inputs):
    prediction = model.predict([inputs])[0]
    predprop = model.predprop([inputs])
    confidence = np.max(predprop)
    return prediction, confidence

# Create a questionnaire for input
user_input = {}
for key, question in questions.items():
    user_input[key] = st.radio(question, [1, 2, 3, 4, 5], index=2)

# Button to trigger the prediction
if st.button("Calculate"):
    prediction, confidence = predict_purchase_intention(list(user_input.values()))
    st.write(f"**Predicted Purchase Intention:** {prediction}")
    st.write(f"**Confidence Level:** {confidence:.2f}")
