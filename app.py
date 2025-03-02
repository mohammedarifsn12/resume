import streamlit as st
import os
import asyncio
from dotenv import load_dotenv
from groq import Groq

# 🔹 Disable Streamlit’s Watcher to prevent auto-reload issues
os.makedirs(os.path.expanduser("~/.streamlit"), exist_ok=True)
with open(os.path.expanduser("~/.streamlit/config.toml"), "w") as f:
    f.write("[server]\nrunOnSave = false\n")

# 🔹 Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("🚨 Error: GROQ_API_KEY not found. Please set it in the .env file.")
    st.stop()

# 🔹 Initialize Groq API Client
client = Groq(api_key=GROQ_API_KEY)

# 🔹 Fix asyncio issue in Streamlit
def get_groq_response(prompt):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(
            client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[{"role": "user", "content": prompt}]
            )
        )
        return response.choices[0].message.content if response.choices else "No response."
    except Exception as e:
        return f"❌ Error: {str(e)}"
    finally:
        loop.close()

# 🔹 Streamlit UI
st.title("🚀 Groq API Chatbot")
st.write("Enter a prompt and get a response from the Groq AI!")

prompt = st.text_input("💬 Enter your prompt:")
if st.button("Generate Response"):
    if prompt.strip():
        st.info("⏳ Generating response, please wait...")
        result = get_groq_response(prompt)
        st.write("🧠 **Groq Response:**")
        st.success(result)
    else:
        st.warning("⚠️ Please enter a prompt before clicking the button.")

