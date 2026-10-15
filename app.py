import joblib
import streamlit as st


@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


model = load_model()

st.title("News Headline Classifier")
headline = st.text_input("Enter a news headline")

if headline:
    probs = model.predict_proba([headline])[0]
    best = probs.argmax()
    prediction = model.classes_[best]
    confidence = probs[best]

    st.write(f"Predicted category: **{prediction}**")
    st.write(f"Confidence: {confidence:.0%}")
    st.progress(float(confidence))

    if confidence < 0.5:
        st.warning(
            "Not sure: this headline could belong to more than one category.")

    st.subheader("Top 3 categories")
    top3 = probs.argsort()[::-1][:3]
    for i in top3:
        st.write(f"{model.classes_[i]}: {probs[i]:.0%}")
