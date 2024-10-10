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
This application assists you in understanding how your shopping preferences influence your purchasing intention.
Please take a moment to select the option that best represents your thoughts for each question below. 
Upon completing your selections, please click the 'Predict' button to view your purchase intention score, where 1 indicates a low likelihood of purchasing from this store and 5 indicates a high likelihood. A confidence score will be displayed to provide the accuracy of the prediction.
""")

# Descriptions for radio buttons
questions = {
    "If prices increase, how likely are you to buy from the store?": [
        (1, "Absolutely won’t buy if prices increase."),
        (2, "Highly unlikely to buy if prices increase."),
        (3, "Undecided about buying if prices increase."),
        (4, "Might buy, but would prefer a lower price."),
        (5, "Will buy even if prices increase."),
    ],
    "If another grocery store offers cheaper prices, how likely are you to buy from the competitor?": [
        (1, "Will definitely buy from a competitor who is cheaper."),
        (2, "Will probably buy from a competitor who is cheaper."),
        (3, "Unsure where to buy goods."),
        (4, "Prefer to buy at the same grocery store, but would consider others."),
        (5, "Will buy at the same grocery store even if prices increase."),
    ],
    "How would you describe the cleanliness of the shopping environment in this store?": [
        (1, "Very dirty shopping environment."),
        (2, "Somewhat dirty shopping environment."),
        (3, "Needs some cleaning but is not overly dirty."),
        (4, "Generally clean but could be improved."),
        (5, "Clean shopping environment."),
    ],
    "How would you rate the overall quality of the products in the store?": [
        (1, "Overall quality of the products is very poor."),
        (2, "Overall quality of the products is poor."),
        (3, "Overall quality of the products is average."),
        (4, "Overall quality of the products is good."),
        (5, "Overall quality of the products is excellent."),
    ],
    "How would you rate the quality of the fresh produce in the store?": [
        (1, "Quality of the produce is very poor."),
        (2, "Quality of the produce is poor."),
        (3, "Quality of the produce is average."),
        (4, "Quality of the produce is good."),
        (5, "Quality of the produce is excellent."),
    ],
    "How would you rate the quality of the meat department in the store?": [
        (1, "Quality of the meat department is very poor."),
        (2, "Quality of the meat department is poor."),
        (3, "Quality of the meat department is average."),
        (4, "Quality of the meat department is good."),
        (5, "Quality of the meat department is excellent."),
    ],
    "How well does the store meet your needs?": [
        (1, "Store does not meet my needs at all."),
        (2, "Store rarely meets my needs."),
        (3, "Store sometimes meets my needs."),
        (4, "Store usually meets my needs."),
        (5, "Store always meets my needs."),
    ],
    "How consistent is the store in providing good quality products?": [
        (1, "Store does not consistently provide good quality products."),
        (2, "Store rarely provides good quality products consistently."),
        (3, "Store sometimes provides good quality products consistently."),
        (4, "Store usually provides good quality products."),
        (5, "Store consistently provides good quality products."),
    ],
    "How would you rate the value for money of the products in the store?": [
        (1, "Products are not value for money at all."),
        (2, "Products are generally not value for money."),
        (3, "Products are sometimes value for money."),
        (4, "Products offer reasonable value for money."),
        (5, "Products are good value for money."),
    ],
    "How affordable are the products in the store?": [
        (1, "Products are not affordable at all."),
        (2, "Products are generally not affordable."),
        (3, "Products range from being affordable to expensive."),
        (4, "Products are generally affordable."),
        (5, "Products are very affordable."),
    ],
    "How much do you save money by shopping at this store compared to others?": [
        (1, "Will definitely save money at the competitor store."),
        (2, "Will probably save money at the competitor store."),
        (3, "Indecisive about which shop provides opportunities to save money."),
        (4, "May save money at this grocery store."),
        (5, "Will save money at this grocery store."),
    ],
}

# Collect responses
responses = {}
for question, options in questions.items():
    st.markdown(f"### {question}")  # Larger font for questions
    selected_value = st.radio(
        "",
        [opt[1] for opt in options],
        index=0
    )
    # Extract the numerical value from the selected response
    value = int(options[[opt[1] for opt in options].index(selected_value)][0])  # Get the number from the tuple
    responses[question] = value

# Input data to be used for prediction
input_data = list(responses.values())

# Predict when the button is pressed
if st.button("Predict"):
    prediction, confidence_interval = predict_purchase_intention(input_data)
    
    st.write(f"### Predicted Purchase Intention: {prediction}")
    st.write(f"### Confidence Interval: {confidence_interval:.2f}%")
