import streamlit as st
import requests

st.set_page_config(page_title="Legal Chatbot", layout="centered")
st.title("⚖️ LawBot: Legal Case Analyzer")

with st.form("legal_case_form"):
    description = st.text_area("Enter the legal case description", height=200)
    submitted = st.form_submit_button("Analyze Case")

if submitted and description.strip():
    with st.spinner("Analyzing case..."):
        response = requests.post("http://localhost:8000/analyze/", json={"description": description})
        if response.status_code == 200:
            data = response.json()
            st.success(f"**Predicted Case Type**: {data['predicted_case_type'].capitalize()}")

            st.markdown("### 📊 Case Type Confidence Scores")
            st.bar_chart(data["case_type_scores"])

            st.markdown("### 📌 Predicted IPC Sections")
            st.code(data["ipc_sections"], language="text")
        else:
            st.error("Failed to analyze case. Please try again.")
