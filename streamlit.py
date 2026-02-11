import streamlit as st
import requests
import json

st.title("Sugar-AI Chat Interface")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Add API key field
api_key = st.sidebar.text_input("API Key", type="password")

# Endpoint selection (moved to sidebar)
endpoint_choice = st.sidebar.selectbox(
    "Choose endpoint:",
    ["RAG (ask)", "Direct LLM (ask-llm)", "Custom Prompt (ask-llm-prompted)"],
)

st.subheader("Ask Sugar-AI")

# Custom prompt section for ask-llm-prompted
custom_prompt = ""
generation_params = {}

if endpoint_choice == "Custom Prompt (ask-llm-prompted)":
    custom_prompt = st.text_area(
        "Custom System Prompt:",
        value="You are a helpful AI assistant. Answer the question concisely.",
    )
    st.subheader("Generation Parameters")

    col1, col2 = st.columns(2)
    with col1:
        max_new_tokens = st.number_input(
            "Max New Tokens", min_value=1, max_value=2048, value=256
        )
        temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=0.7)
        repetition_penalty = st.slider(
            "Repetition Penalty", min_value=1.0, max_value=2.0, value=1.1
        )

    with col2:
        top_p = st.slider("Top P", min_value=0.1, max_value=1.0, value=0.9)
        top_k = st.number_input("Top K", min_value=1, max_value=100, value=50)

    generation_params = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "top_p": top_p,
        "top_k": top_k,
    }

if question := st.chat_input("Enter your question"):
    if not api_key:
        st.warning("Please enter an API key.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(question)

        headers = {"X-API-Key": api_key}

        try:
            if endpoint_choice == "RAG (ask)":
                url = "http://localhost:8000/ask"
                params = {"question": question}
                response = requests.post(url, params=params, headers=headers)

            elif endpoint_choice == "Direct LLM (ask-llm)":
                url = "http://localhost:8000/ask-llm"
                params = {"question": question}
                response = requests.post(url, params=params, headers=headers)

            elif endpoint_choice == "Custom Prompt (ask-llm-prompted)":
                url = "http://localhost:8000/ask-llm-prompted"
                headers["Content-Type"] = "application/json"
                data = {
                    "question": question,
                    "custom_prompt": custom_prompt,
                    **generation_params,
                }
                response = requests.post(url, headers=headers, data=json.dumps(data))

            if response.status_code == 200:
                result = response.json()
                answer = result["answer"]

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})

                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.markdown(answer)

                st.sidebar.info(f"User: {result.get('user', 'Unknown')}")
                st.sidebar.info(
                    f"Remaining quota: {result['quota']['remaining']}/{result['quota']['total']}"
                )

                # Show generation parameters for custom prompt endpoint
                if (
                    endpoint_choice == "Custom Prompt (ask-llm-prompted)"
                    and "generation_params" in result
                ):
                    with st.expander("Generation Parameters Used"):
                        st.json(result["generation_params"])

            else:
                st.error(f"Error {response.status_code}: {response.text}")

        except Exception as e:
            st.error(f"Error contacting the API: {e}")
