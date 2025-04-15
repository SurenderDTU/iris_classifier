import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
iris = load_iris()

# App title
st.title("🌸 Iris Flower Classifier")

# Inputs
sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 0.2)

# Classifier
model = RandomForestClassifier()
model.fit(iris.data, iris.target)

# Predict
features = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(features)
species = iris.target_names[prediction[0]]

# Output
st.success(f"Predicted species: **{species}**")
