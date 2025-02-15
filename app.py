import streamlit as st
import PyPDF2
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF

# Set up Gemini AI API Key
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Load pre-trained Sentence Transformer model for job matching
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

# Function to calculate match score between resume and job description
def calculate_match(resume_text, job_desc):
    resume_embedding = model.encode([resume_text])
    job_embedding = model.encode([job_desc])
    similarity_score = cosine_similarity(resume_embedding, job_embedding)[0][0] * 100
    return similarity_score

# Function to get resume improvement suggestions using Gemini AI
def get_resume_improvements(resume_text, job_desc):
    prompt = f"""
    Here is a candidate's resume:
    {resume_text}

    The candidate is applying for the following job:
    {job_desc}

    Please suggest improvements to make the resume ATS-friendly. Highlight missing skills, weak points, and best formatting practices.
    """
    
    response = genai.chat(messages=[{"role": "user", "content": prompt}])
    return response.text if response else "No suggestions available."

# Function to rewrite resume in ATS-friendly format using Gemini AI
def rewrite_ats_resume(resume_text, job_desc):
    prompt = f"""
    Here is a candidate's resume:
    {resume_text}

    The candidate is applying for the following job:
    {job_desc}

    Rewrite the resume in an ATS-friendly format. Use clear section headings (Work Experience, Skills, Education, etc.), bullet points, and simple formatting for easy parsing.
    """

    response = genai.chat(messages=[{"role": "user", "content": prompt}])
    return response.text if response else "No optimized resume available."

# Function to create a properly formatted ATS-friendly resume PDF
def create_ats_pdf(text, filename="ATS_Optimized_Resume.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    
    for line in text.split("\n"):
        if line.strip():
            if ":" in line:  # Formatting section headings
                pdf.set_font("Arial", style="B", size=12)
                pdf.cell(200, 8, txt=line, ln=True, align='L')
                pdf.set_font("Arial", size=11)
            else:
                pdf.cell(200, 8, txt=f"• {line}", ln=True, align='L')
    
    pdf.output(filename)

# Streamlit UI
st.title("📄 AI Resume Matchmaking System (Powered by Gemini AI)")
st.write("Upload your resume and paste a job description to check compatibility, get ATS-friendly improvement suggestions, and download an optimized resume.")

# Upload Resume PDF
uploaded_file = st.file_uploader("Upload Your Resume (PDF)", type=["pdf"])

# Input Job Description
job_desc = st.text_area("Paste Job Description", placeholder="Enter job description here...")

if uploaded_file is not None:
    resume_text = extract_text_from_pdf(uploaded_file)
    st.text_area("Extracted Resume Text", resume_text, height=200)

    if st.button("Find Match Score & Get ATS Suggestions"):
        if job_desc:
            match_score = calculate_match(resume_text, job_desc)
            st.success(f"✅ Match Score: {match_score:.2f}%")

            # Get improvement suggestions
            st.subheader("🔹 Resume Improvement Suggestions")
            suggestions = get_resume_improvements(resume_text, job_desc)
            st.write(suggestions)

            # Auto Rewrite ATS-Friendly Resume
            st.subheader("🔹 ATS-Optimized Resume")
            rewritten_resume = rewrite_ats_resume(resume_text, job_desc)
            st.text_area("ATS-Friendly Resume", rewritten_resume, height=300)

            # Download as PDF
            if st.button("Download ATS-Friendly Resume as PDF"):
                create_ats_pdf(rewritten_resume)
                with open("ATS_Optimized_Resume.pdf", "rb") as file:
                    st.download_button(label="📥 Download Resume", data=file, file_name="ATS_Optimized_Resume.pdf", mime="application/pdf")

        else:
            st.warning("⚠ Please enter the job description.")
