import streamlit as st
import joblib

# Load model (make sure model.pkl is in repo)
model = joblib.load("model.pkl")

# Page config
st.set_page_config(page_title="SMS Spam Detector", page_icon="📩")

# Title
st.title("📩 SMS Spam Detection App")

st.write("Enter a message below to check whether it is Spam or Not Spam.")

# Input box
message = st.text_area("Enter SMS message")

# Predict button
if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message!")
    else:
        prediction = model.predict([message])

        if prediction[0] == 1:
            st.error("🚨 Spam Message Detected!")
        else:
            st.success("✅ This is a Normal Message")

# Footer
st.markdown("---")
st.caption("Built with Streamlit 🚀")
if file:
    df = pd.read_csv(file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

# Footer
st.markdown("---")
st.caption("🚀 Deployed using Streamlit Cloud")
