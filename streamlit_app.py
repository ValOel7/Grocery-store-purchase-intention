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
    confidence_interval = np.max(confidence) * 100  # Get the highest probability as confidence
    return prediction, confidence_interval

# Streamlit app
st.title("Purchase Intention Prediction App")

st.write("""
Please answer the following questions by selecting a value (between 1 and 5):
""")

# Function to display questions with info button
def display_question_with_info(question, info_text):
    col1, col2 = st.columns([4, 1])  # Create two columns for question and info button
    with col1:
        response = st.radio(question, [1, 2, 3, 4, 5])
    with col2:
        if st.button("ℹ️", key=question):  # Info button with a unique key
            st.write(info_text)  # Display the info text when button is clicked
    return response

# Questions with their corresponding info texts
PS1_info = "1: Won’t buy if prices increase.  2: Won’t buy if prices increase.  3: Unsure to buy or not if prices increase.  4: Will buy even if prices increase.  5: Will buy even if prices increase."
PS1 = display_question_with_info("If prices increase, how likely are you to buy from the store?", PS1_info)

PS3_info = "1: Will buy from cheaper competitor.  2: Will buy from cheaper competitors.  3: Unsure at which grocery store to buy goods.  4: Will buy at same grocery store even if prices increase.  5: Will buy at same grocery store even if prices increase."
PS3 = display_question_with_info("If another grocery store offers cheaper prices, how likely are you to buy from the competitor?", PS3_info)

PE2_info = "1: Dirty shopping environment.  2: Dirty shopping environment.  3: Not a dirty shopping environment but needs some cleaning.  4: Clean shopping environment.  5: Clean shopping environment."
PE2 = display_question_with_info("How would you describe the cleanliness of the shopping environment in this store?", PE2_info)

PPQ1_info = "1: Overall quality of the products is poor.  2: Overall quality of the products is poor.  3: Overall quality of the products is average.  4: Overall quality of the products is good.  5: Overall quality of the products is good."
PPQ1 = display_question_with_info("How would you rate the overall quality of the products in the store?", PPQ1_info)

PPQ2_info = "1: Quality of the produce is poor.  2: Quality of the produce is poor.  3: Quality of the produce is average.  4: Quality of the produce is good.  5: Quality of the produce is good."
PPQ2 = display_question_with_info("How would you rate the quality of the fresh produce in the store?", PPQ2_info)

PPQ3_info = "1: Quality of meat department is poor.  2: Quality of meat department is poor.  3: Quality of meat department is average.  4: Quality of meat department is good.  5: Quality of meat department is good."
PPQ3 = display_question_with_info("How would you rate the quality of the meat department in the store?", PPQ3_info)

CT2_info = "1: Store does not meet my needs.  2: Store does not meet my needs.  3: Store sometimes meets my needs.  4: Store always meets my needs.  5: Store always meets my needs."
CT2 = display_question_with_info("How well does the store meet your needs?", CT2_info)

CT5_info = "1: Store does not consistently provide good quality products.  2: Store does not consistently provide good quality products.  3: Store sometimes provides good quality products consistently.  4: Store consistently provides good quality products.  5: Store consistently provides good quality products."
CT5 = display_question_with_info("How consistent is the store in providing good quality products?", CT5_info)

PV1_info = "1: Products are not value for money.  2: Products are not value for money.  3: Products are sometimes value for money.  4: Products are good value for money.  5: Products are good value for money."
PV1 = display_question_with_info("How would you rate the value for money of the products in the store?", PV1_info)

PV2_info = "1: Products are not affordable.  2: Products are not affordable.  3: Products range from being affordable to expensive.  4: Products are affordable.  5: Products are affordable."
PV2 = display_question_with_info("How affordable are the products in the store?", PV2_info)

PV3_info = "1: Save money at the competitor store.  2: Save money at the competitor store.  3: Indecisive about which shop provides an opportunity to save money.  4: Save money at this grocery store.  5: Save money at this grocery store."
PV3 = display_question_with_info("How much do you save money by shopping at this store compared to others?", PV3_info)

# Input data to be used for prediction
input_data = [PS1, PS3, PE2, PPQ1, PPQ2, PPQ3, CT2, CT5, PV1, PV2, PV3]

# Predict when the button is pressed
if st.button("Predict"):
    prediction, confidence_interval = predict_purchase_intention(input_data)
    
    st.write(f"### Predicted Purchase Intention: {prediction}")
    st.write(f"### Confidence Interval: {confidence_interval:.2f}%")
