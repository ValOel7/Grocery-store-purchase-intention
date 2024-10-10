import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open('rf.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to predict purchase intention
def predict_purchase_intention(input_data):
    prediction = model.predict([input_data])[0]
    # Get confidence probabilities (optional)
    confidence = model.predict_proba([input_data])[0]
    confidence_interval = np.max(confidence) * 100  # Get the highest probability as confidence
    return prediction, confidence_interval

# Streamlit app
st.title("Purchase Intention Prediction App")

st.write("""
Please answer the following questions by selecting a value (between 1 and 5):
""")

# Input form using questions with radio buttons for each variable
PS1 = st.radio("If prices increase, how likely are you to buy from the store?", [1, 2, 3, 4, 5])
PS3 = st.radio("If another grocery store offers cheaper prices, how likely are you to buy from the competitor?", [1, 2, 3, 4, 5])
PE2 = st.radio("How would you describe the cleanliness of the shopping environment in this store?", [1, 2, 3, 4, 5])
PPQ1 = st.radio("How would you rate the overall quality of the products in the store?", [1, 2, 3, 4, 5])
PPQ2 = st.radio("How would you rate the quality of the fresh produce in the store?", [1, 2, 3, 4, 5])
PPQ3 = st.radio("How would you rate the quality of the meat department in the store?", [1, 2, 3, 4, 5])
CT2 = st.radio("How well does the store meet your needs?", [1, 2, 3, 4, 5])
CT5 = st.radio("How consistent is the store in providing good quality products?", [1, 2, 3, 4, 5])
PV1 = st.radio("How would you rate the value for money of the products in the store?", [1, 2, 3, 4, 5])
PV2 = st.radio("How affordable are the products in the store?", [1, 2, 3, 4, 5])
PV3 = st.radio("How much do you save money by shopping at this store compared to others?", [1, 2, 3, 4, 5])

# Input data to be used for prediction
input_data = [PS1, PS3, PE2, PPQ1, PPQ2, PPQ3, CT2, CT5, PV1, PV2, PV3]

# Predict when the button is pressed
if st.button("Predict"):
    prediction, confidence_interval = predict_purchase_intention(input_data)
    
    st.write(f"### Predicted Purchase Intention: {prediction}")
    st.write(f"### Confidence Interval: {confidence_interval:.2f}%")

