
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import base64
from sklearn.model_selection import train_test_split
from datetime import datetime


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('ss.jpg')  

 
df = pd.read_csv('train.csv')

X_train, X_test, y_train, y_test = train_test_split(df[['store', 'item', 'date']], df['sales'], test_size=0.2, random_state=42)
model = DecisionTreeRegressor()
model.fit(X_train[['store', 'item']], y_train)

accuracy = model.score(X_test[['store', 'item']], y_test)
print("model used:",model)
print("r2:", accuracy)

def make_prediction(date, store, item):
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    prediction = model.predict([[store, item, date_obj]])
    return prediction[0]

import streamlit as st

# Create the input form
st.write("# Sales Prediction App")
date = st.date_input("Select the date")
store = st.number_input("Enter the store number", min_value=1, max_value=10)
item = st.number_input("Enter the item number", min_value=1, max_value=50)

# Make the prediction
if st.button("Predict"):
    prediction = make_prediction(date, store, item)
    st.write("The predicted sales for {} at store {} for item {} is {}".format(date, store, item, prediction))
