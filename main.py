import streamlit as st
import pandas as pd

# Page settings
st.set_page_config(page_title="My Online App", page_icon="🌐")

# Title
st.title("🌐 My First Online Streamlit App")

# Input
name = st.text_input("Enter your name")

if name:
    st.success(f"Hello {name} 👋 Welcome to your online app!")

# Simple selectbox
choice = st.selectbox("Choose a field", ["Machine Learning", "Web Dev", "Python"])

st.write("You selected:", choice)

# Button
if st.button("Click Me"):
    st.info("Button clicked successfully!")

# Sample Data
st.subheader("📊 Sample Data")
data = pd.DataFrame({
    "Name": ["Dev", "AI", "ML"],
    "Score": [90, 85, 88]
})

st.dataframe(data)

# File Upload
st.subheader("📂 Upload CSV File")
file = st.file_uploader("Upload your file")

if file:
    df = pd.read_csv(file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

# Footer
st.markdown("---")
st.caption("🚀 Deployed using Streamlit Cloud")