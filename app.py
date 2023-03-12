
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

# Load the dataset
df = pd.read_csv('train.csv')

# Preprocess the date column
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[['store', 'item', 'year', 'month', 'day', 'day_of_week']], df['sales'], test_size=0.2, random_state=42)

# Train a machine learning model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print("model used:",model)
print("r2:", accuracy)

# Define a function to make predictions
def make_prediction(date, store, item):
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    year = date_obj.year
    month = date_obj.month
    day = date_obj.day
    day_of_week = date_obj.weekday()
    prediction = model.predict([[store, item, year, month, day, day_of_week]])
    return prediction[0]

# Create the input form
st.write("# Sales Prediction App")
date = st.date_input("Select the date")
store = st.number_input("Enter the store number", min_value=1, max_value=10)
item = st.number_input("Enter the item number", min_value=1, max_value=50)

# Make the prediction
if st.button("Predict"):
    prediction = make_prediction(date, store, item)
    st.write("The predicted sales for {} at store {} for item {} is {}".format(date, store, item, prediction))
