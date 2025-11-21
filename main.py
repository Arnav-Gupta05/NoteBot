import streamlit as st
import os
import PyPDF2
import docx
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")


# -------------------------------
# File Loading Helpers
# -------------------------------
def load_txt(file):
    return file.read().decode("utf-8")

def load_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def load_docx(file):
    d = docx.Document(file)
    return "\n".join([p.text for p in d.paragraphs])

def extract_text(file):
    if file.type == "text/plain":
        return load_txt(file)
    elif file.type == "application/pdf":
        return load_pdf(file)
    elif file.type in (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword"
    ):
        return load_docx(file)
    else:
        return ""


# -------------------------------
# Vector Store with Chroma + MiniLM
# -------------------------------
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = Chroma.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    return vector_store


# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📘 Student Notes Helper Bot ")
st.write("Upload notes (PDF, DOCX, TXT) and ask questions to get simple exam-ready answers.")

uploaded_files = st.file_uploader(
    "Upload your study files",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True
)


if not API_KEY:
    st.error(" GEMINI_API_KEY missing in .env file!")
    st.stop()


if uploaded_files:
    # Extract text
    all_text = ""
    for file in uploaded_files:
        all_text += extract_text(file) + "\n"

    st.success("Files uploaded and text extracted!")

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(all_text)

    # Create vector store
    vectorstore = get_vectorstore(chunks)

    # Load Gemini for answering
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=API_KEY)

    # Prompt template
    prompt_template = ChatPromptTemplate.from_template("""
You are **NoteBot**, an advanced study assistant.

Your job is to answer the student’s question using:
1. **The uploaded notes (context)** as the main reference.
2. **Your own knowledge** to add clarity, deeper insights, simple explanations, and useful examples.

### How you must answer:
- First, understand the student’s question.
- Use the **context from the notes** as the factual base.
- Add **your own knowledge** to fill gaps, simplify ideas, and make explanations clearer.
- Break down complex concepts into **very simple steps**.
- Whenever needed, give a **short exam-oriented summary**.
- Make the tone friendly, clear, and student-focused.
- Do NOT contradict the notes. If notes are incomplete, expand based on general knowledge.

---

### 📘 Uploaded Notes (Context):
{context}

### ❓ Student Question:
{question}

---

### 🧠 Format your answer like this:
- **Simple Explanation:** (easy overview)
- **Detailed Breakdown:** (steps, components, or points)
- **Examples / Analogies:** (very easy to understand)
- **Exam-Ready Summary:** (3–5 lines)

Begin your answer:
""")

    # Question input
    question = st.text_input("Ask your question from the notes:")

    if question:
        docs = vectorstore.similarity_search(question, k=4)
        context = "\n".join([d.page_content for d in docs])

        prompt = prompt_template.format(context=context, question=question)
        response = llm.invoke(prompt)

        st.subheader("Answer:")
        st.write(response.content)

else:
    st.info("Upload files to begin.")
