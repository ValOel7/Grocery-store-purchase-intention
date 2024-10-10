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

# Function to create a radio button with info tooltip
def display_question_with_info(question, info):
    col1, col2 = st.columns([8, 1])  # Create two columns
    with col1:
        response = st.radio(question, [1, 2, 3, 4, 5])
    with col2:
        if st.button("ℹ️", key=question):  # Use the "ℹ️" emoji as the info button
            st.markdown(f"<div style='font-size: 12px;'>{info}</div>", unsafe_allow_html=True)
    return response

# Define updated information texts with Markdown formatting
PS1_info = """
1: Absolutely won’t buy if prices increase.<br>
2: Highly unlikely to buy if prices increase.<br>
3: Undecided about buying if prices increase.<br>
4: Might buy, but would prefer a lower price.<br>
5: Will buy even if prices increase.
"""

PS3_info = """
1: Will definitely buy from a competitor who is cheaper.<br>
2: Will probably buy from a competitor who is cheaper.<br>
3: Unsure where to buy goods.<br>
4: Prefer to buy at the same grocery store, but would consider others.<br>
5: Will buy at the same grocery store even if prices increase.
"""

PE2_info = """
1: Very dirty shopping environment.<br>
2: Somewhat dirty shopping environment.<br>
3: Needs some cleaning but is not overly dirty.<br>
4: Generally clean but could be improved.<br>
5: Clean shopping environment.
"""

PPQ1_info = """
1: Overall quality of the products is very poor.<br>
2: Overall quality of the products is poor.<br>
3: Overall quality of the products is average.<br>
4: Overall quality of the products is good.<br>
5: Overall quality of the products is excellent.
"""

PPQ2_info = """
1: Quality of the produce is very poor.<br>
2: Quality of the produce is poor.<br>
3: Quality of the produce is average.<br>
4: Quality of the produce is good.<br>
5: Quality of the produce is excellent.
"""

PPQ3_info = """
1: Quality of the meat department is very poor.<br>
2: Quality of the meat department is poor.<br>
3: Quality of the meat department is average.<br>
4: Quality of the meat department is good.<br>
5: Quality of the meat department is excellent.
"""

CT2_info = """
1: Store does not meet my needs at all.<br>
2: Store rarely meets my needs.<br>
3: Store sometimes meets my needs.<br>
4: Store usually meets my needs.<br>
5: Store always meets my needs.
"""

CT5_info = """
1: Store does not consistently provide good quality products.<br>
2: Store rarely provides good quality products consistently.<br>
3: Store sometimes provides good quality products consistently.<br>
4: Store usually provides good quality products.<br>
5: Store consistently provides good quality products.
"""

PV1_info = """
1: Products are not value for money at all.<br>
2: Products are generally not value for money.<br>
3: Products are sometimes value for money.<br>
4: Products offer reasonable value for money.<br>
5: Products are good value for money.
"""

PV2_info = """
1: Products are not affordable at all.<br>
2: Products are generally not affordable.<br>
3: Products range from being affordable to expensive.<br>
4: Products are generally affordable.<br>
5: Products are very affordable.
"""

PV3_info = """
1: Will definitely save money at the competitor store.<br>
2: Will probably save money at the competitor store.<br>
3: Indecisive about which shop provides opportunities to save money.<br>
4: May save money at this grocery store.<br>
5: Will save money at this grocery store.
"""

# Input form using questions with radio buttons for each variable
PS1 = display_question_with_info("If prices increase, how likely are you to buy from the store?", PS1_info)
PS3 = display_question_with_info("If another grocery store offers cheaper prices, how likely are you to buy from the competitor?", PS3_info)
PE2 = display_question_with_info("How would you describe the cleanliness of the shopping environment in this store?", PE2_info)
PPQ1 = display_question_with_info("How would you rate the overall quality of the products in the store?", PPQ1_info)
PPQ2 = display_question_with_info("How would you rate the quality of the fresh produce in the store?", PPQ2_info)
PPQ3 = display_question_with_info("How would you rate the quality of the meat department in the store?", PPQ3_info)
CT2 = display_question_with_info("How well does the store meet your needs?", CT2_info)
CT5 = display_question_with_info("How consistent is the store in providing good quality products?", CT5_info)
PV1 = display_question_with_info("How would you rate the value for money of the products in the store?", PV1_info)
PV2 = display_question_with_info("How affordable are the products in the store?", PV2_info)
PV3 = display_question_with_info("How much do you save money by shopping at this store compared to others?", PV3_info)

# Input data to be used for prediction
input_data = [PS1, PS3, PE2, PPQ1, PPQ2, PPQ3, CT2, CT5, PV1, PV2, PV3]

# Predict when the button is pressed
if st.button("Predict"):
    prediction, confidence_interval = predict_purchase_intention(input_data)
    
    st.write(f"### Predicted Purchase Intention: {prediction}")
    st.write(f"### Confidence Interval: {confidence_interval:.2f}%")
