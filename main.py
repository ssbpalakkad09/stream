import pickle
import streamlit as st
import numpy as np

st.title("Flower Classification App")

file_name = "model.pkl"
with open("model.pkl", 'rb') as file:
    model = pickle.load(file)

s1 = st.slider("Sepal Length", 0.0, 10.0, 5.0)
s2 = st.slider("Sepal Width", 0.0, 10.0, 3.0)
s3 = st.slider("Petal Length", 0.0, 10.0, 4.0)
s4 = st.slider("Petal Width", 0.0, 10.0, 1.0)

if st.button("Predict"):

    data = np.array([[s1, s2, s3, s4]])
    pred = model.predict(data)
    st.write(f"The predicted flower is: {pred[0]}")
