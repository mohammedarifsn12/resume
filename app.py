import streamlit as st
import os
import asyncio
from dotenv import load_dotenv
from groq import Groq

# Ensure Streamlit config directory exists
os.makedirs(os.path.expanduser("~/.streamlit"), exist_ok=True)
with open(os.path.expanduser("~/.streamlit/config.toml"), "w") as f:
    f.write("[server]\nrunOnSave = false\n")

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ Error: GROQ_API_KEY is missing. Set it in your .env file.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# Async function to fetch response
async def fetch_groq_response(prompt):
    try:
        response = await client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content if response.choices else "No response."
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Wrapper to run async function in Streamlit
def get_groq_response(prompt):
    return asyncio.run(fetch_groq_response(prompt))

# Streamlit UI
st.title("Groq API Chatbot")
st.markdown("### Powered by Mixtral-8x7B-32768")

prompt = st.text_input("Enter your prompt:")
if st.button("Generate Response"):
    with st.spinner("Generating response..."):
        result = get_groq_response(prompt)
    st.write(result)
