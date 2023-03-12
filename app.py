import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
from sklearn.model_selection import train_test_split
df = pd.read_csv('train.csv')
X_train, X_test, y_train, y_test = train_test_split(df[['store', 'item']], df['sales'], test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print("r2:", accuracy)
prediction = model.predict([[store, item]])
print("Prediction:", prediction)
import streamlit as st
# Create the input form
st.write("# Sales Prediction App")
date = st.date_input("Select the date")
store = st.number_input("Enter the store number", min_value=1, max_value=10)
item = st.number_input("Enter the item number", min_value=1, max_value=50)
# Make the prediction
if st.button("Predict"):
    prediction = model.predict([[store, item]])
    st.write("The predicted sales for {} at store {} for item {} is {}".format(date, store, item, prediction))

   
    
    


