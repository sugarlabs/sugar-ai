import streamlit as st
import requests

st.title("Sugar-AI Chat Interface")

use_rag = st.checkbox("Use RAG (Retrieval-Augmented Generation)", value=True)

st.subheader("Ask Sugar-AI")
question = st.text_input("Enter your question:")

if st.button("Submit"):
    if question:
        if use_rag:
            url = "http://localhost:8000/ask"
        else:
            url = "http://localhost:8000/ask-llm"
        params = {"question": question}
        try:
            response = requests.post(url, params=params)
            if response.status_code == 200:
                result = response.json()
                st.markdown("**Answer:** " + result["answer"])
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Error contacting the API: {e}")
    else:
        st.warning("Please enter a question.")
