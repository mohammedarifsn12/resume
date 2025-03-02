import streamlit as st
import PyPDF2
import os
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
import io
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Groq API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    return text

# Function to calculate resume-job match percentage
def calculate_match(resume_text, job_desc):
    resume_embedding = model.encode([resume_text])
    job_embedding = model.encode([job_desc])
    similarity_score = cosine_similarity(resume_embedding, job_embedding)[0][0] * 100
    return similarity_score

# Function to interact with Groq API
def get_groq_response(prompt, model="llama3-8b"):
    url = "https://api.groq.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response.")
    else:
        return f"Error: {response.text}"

# Function to get AI-powered resume improvement suggestions
def get_resume_improvements(resume_text, job_desc):
    prompt = f"""
    Here is a candidate's resume:
    {resume_text}

    The candidate is applying for the following job:
    {job_desc}

    Please suggest improvements to make the resume ATS-friendly. Highlight missing skills, weak points, and best formatting practices.
    """
    return get_groq_response(prompt)

# Function to rewrite resume in an ATS-friendly format
def rewrite_ats_resume(resume_text, job_desc):
    prompt = f"""
    Here is a candidate's resume:
    {resume_text}

    The candidate is applying for the following job:
    {job_desc}

    Rewrite the resume in an ATS-friendly format. Use proper headings (Work Experience, Skills, Education, etc.), bullet points, and clear formatting for easy parsing.
    """
    return get_groq_response(prompt)

@st.cache_data  # Cache the rewritten resume
def get_rewritten_resume(resume_text, job_desc):
    return rewrite_ats_resume(resume_text, job_desc)

# Function to create ATS-optimized PDF
def create_ats_pdf(text):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=11)

    for line in text.splitlines():
        if line.strip():
            if ":" in line:
                pdf.set_font("Arial", style="B", size=12)
                pdf.cell(200, 8, txt=line, ln=True, align='L')
                pdf.set_font("Arial", size=11)
            else:
                pdf.cell(200, 8, txt=f"• {line}", ln=True, align='L')

    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)
    pdf_bytes = pdf_buffer.getvalue()
    return pdf_bytes

# Streamlit UI
st.title("📄 AI Resume Matchmaking System (ATS-Friendly)")

uploaded_file = st.file_uploader("Upload Your Resume (PDF)", type=["pdf"])
job_desc = st.text_area("Paste Job Description", placeholder="Enter job description here...")

if uploaded_file is not None:
    try:
        resume_text = extract_text_from_pdf(uploaded_file)
        st.text_area("Extracted Resume Text", resume_text, height=200)

        if st.button("Find Match Score & Get ATS Suggestions"):
            if job_desc:
                match_score = calculate_match(resume_text, job_desc)
                st.success(f"✅ Match Score: {match_score:.2f}%")

                st.subheader("🔹 Resume Improvement Suggestions")
                suggestions = get_resume_improvements(resume_text, job_desc)
                st.write(suggestions)

                st.subheader("🔹 ATS-Optimized Resume")
                rewritten_resume = get_rewritten_resume(resume_text, job_desc)
                st.text_area("ATS-Friendly Resume", rewritten_resume, height=300)

                if st.button("Download ATS-Friendly Resume as PDF"):
                    try:
                        pdf_bytes = create_ats_pdf(rewritten_resume)
                        b64 = base64.b64encode(pdf_bytes).decode()
                        href = f"data:application/pdf;base64,{b64}"
                        st.markdown(f'<a href="{href}" download="ATS_Optimized_Resume.pdf">📥 Download Resume</a>', unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Error during PDF creation/download: {e}")

            else:
                st.warning("⚠ Please enter the job description.")

    except Exception as e:
        st.error(f"Error during PDF processing: {e}")
