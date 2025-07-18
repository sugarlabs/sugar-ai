import streamlit as st
from datetime import datetime

# ------------------- Page Config -------------------
st.set_page_config(
    page_title="Sugar AI Lab",
    page_icon="🧠",
    layout="wide"
)

# ------------------- Styling -------------------
st.markdown("""
    <style>
    .main {
        background-color: #f4f6f8;
    }
    .reportview-container .markdown-text-container {
        font-family: 'Helvetica';
    }
    .title {
        font-size:36px !important;
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- Sidebar -------------------
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("Customize your Sugar AI Lab experience.")
theme = st.sidebar.selectbox("Theme", ["Light", "Dark"])
model = st.sidebar.selectbox("LLM Model", ["OpenAI GPT-4", "LLaMA 3", "Mistral", "Gemini Pro"])

# ------------------- Header -------------------
st.markdown("<h1 class='title'>🧪 Sugar AI Lab Interface</h1>", unsafe_allow_html=True)
st.markdown("Welcome to the Sugar AI Streamlit interface! Use the panel below to interact with the model.")

# ------------------- Chat Input Section -------------------
st.subheader("🧠 Ask Something to Sugar AI")

user_input = st.text_input("Enter your prompt:", placeholder="e.g., Summarize this paper or Write a Dockerfile...")
if st.button("Generate Response"):
    if user_input:
        # Placeholder for LLM Response (You can integrate real API here)
        st.success("✅ Model response placeholder:")
        st.info("Sorry, the LLM backend is not connected yet. But your UI works!")
    else:
        st.warning("Please enter a prompt before submitting.")

# ------------------- History Section -------------------
st.subheader("🕒 History")
st.markdown("*(This section will display your previous interactions once session storage is enabled)*")
# Example static history
with st.expander("View Example History"):
    st.markdown("- **[2025-07-19 03:00]** Summarize this GitHub issue\n- **[2025-07-18 17:12]** What is vector DB?")

# ------------------- Footer -------------------
st.markdown("---")
st.caption(f"🧠 Sugar AI Streamlit UI • {datetime.now().strftime('%B %d, %Y')}")