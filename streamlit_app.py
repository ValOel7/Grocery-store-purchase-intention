import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open('rf.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to predict purchase intention
def predict_purchase_intention(input_data):
    prediction = model.predict([input_data])[0]
    confidence = model.predict_proba([input_data])[0]
    confidence_interval = np.max(confidence) * 100
    return prediction, confidence_interval

# Streamlit app
st.title("Purchase Intention Prediction App")

st.write("Please answer the following questions by selecting a value (between 1 and 5):")

# Function to display radio with custom descriptions
def display_radio_with_description(question, descriptions):
    st.write(question)
    for value, desc in descriptions.items():
        st.write(f"- **{value}**: {desc}")
    return st.radio("Select your response:", list(descriptions.keys()))

# Questions with descriptions
PS1 = display_radio_with_description(
    "If prices increase, how likely are you to buy from the store?",
    {1: "Won’t buy if prices increase.",
     2: "Won’t buy if prices increase.",
     3: "Unsure to buy or not if prices increase.",
     4: "Will buy even if prices increase.",
     5: "Will buy even if prices increase."}
)

PS3 = display_radio_with_description(
    "If another grocery store offers cheaper prices, how likely are you to buy from the competitor?",
    {1: "Will buy from competitor who is cheaper.",
     2: "Will buy from competitor who is cheaper.",
     3: "Unsure at which grocery store to buy goods.",
     4: "Will buy at same grocery store even if prices increase.",
     5: "Will buy at same grocery store even if prices increase."}
)

PE2 = display_radio_with_description(
    "How would you describe the cleanliness of the shopping environment in this store?",
    {1: "Dirty shopping environment.",
     2: "Dirty shopping environment.",
     3: "Not a dirty shopping environment but needs some cleaning.",
     4: "Clean shopping environment.",
     5: "Clean shopping environment."}
)

PPQ1 = display_radio_with_description(
    "How would you rate the overall quality of the products in the store?",
    {1: "Overall quality of the products is poor.",
     2: "Overall quality of the products is poor.",
     3: "Overall quality of the products is average.",
     4: "Overall quality of the products is good.",
     5: "Overall quality of the products is good."}
)

PPQ2 = display_radio_with_description(
    "How would you rate the quality of the fresh produce in the store?",
    {1: "Quality of the produce is poor.",
     2: "Quality of the produce is poor.",
     3: "Quality of the produce is average.",
     4: "Quality of the produce is good.",
     5: "Quality of the produce is good."}
)

PPQ3 = display_radio_with_description(
    "How would you rate the quality of the meat department in the store?",
    {1: "Quality of meat department is poor.",
     2: "Quality of meat department is poor.",
     3: "Quality of meat department is average.",
     4: "Quality of meat department is good.",
     5: "Quality of meat department is good."}
)

CT2 = display_radio_with_description(
    "How well does the store meet your needs?",
    {1: "Store does not meet my needs.",
     2: "Store does not meet my needs.",
     3: "Store sometimes meets my needs.",
     4: "Store always meets my needs.",
     5: "Store always meets my needs."}
)

CT5 = display_radio_with_description(
    "How consistent is the store in providing good quality products?",
    {1: "Store does not consistently provide good quality products.",
     2: "Store does not consistently provide good quality products.",
     3: "Store sometimes provides good quality products consistently.",
     4: "Store consistently provides good quality products.",
     5: "Store consistently provides good quality products."}
)

PV1 = display_radio_with_description(
    "How would you rate the value for money of the products in the store?",
    {1: "Products are not value for money.",
     2: "Products are not value for money.",
     3: "Products are sometimes value for money.",
     4: "Products are good value for money.",
     5: "Products are good value for money."}
)

PV2 = display_radio_with_description(
    "How affordable are the products in the store?",
    {1: "Products are not affordable.",
     2: "Products are not affordable.",
     3: "Products range from being affordable to expensive.",
     4: "Products are affordable.",
     5: "Products are affordable."}
)

PV3 = display_radio_with_description(
    "How much do you save money by shopping at this store?",
    {1: "Save money at the competitor store.",
     2: "Save money at the competitor store.",
     3: "Indecisive about which shop provides an opportunity to save money.",
     4: "Save money at this grocery store.",
     5: "Save money at this grocery store."}
)

# Input data preparation
input_data = [PS1, PS3, PE2, PPQ1, PPQ2, PPQ3, CT2, CT5, PV1, PV2, PV3]

# Prediction button
if st.button("Predict Purchase Intention"):
    prediction, confidence_interval = predict_purchase_intention(input_data)
    st.write(f"Predicted Purchase Intention: {prediction}")
    st.write(f"Confidence Interval: {confidence_interval:.2f}%")
