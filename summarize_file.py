import streamlit as st
import pdfplumber
from PIL import Image
import pytesseract
from transformers import pipeline
import io

# Function to extract text from an image
def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text

# Function to extract text from a PDF
def extract_text_from_pdf(pdf):
    text = ""
    with pdfplumber.open(pdf) as pdf_file:
        for page in pdf_file.pages:
            text += page.extract_text()
    return text

# Function to summarize text
def summarize_text(text):
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Streamlit app
def main():
    st.title("Text Extractor and Summarizer")

    uploaded_file = st.file_uploader("Upload an Image or PDF", type=["jpg", "jpeg", "png", "pdf"])

    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            with st.spinner("Extracting text from PDF..."):
                text = extract_text_from_pdf(uploaded_file)
        else:
            image = Image.open(uploaded_file)
            with st.spinner("Extracting text from image..."):
                text = extract_text_from_image(image)

        st.subheader("Extracted Text")
        st.write(text)

        if text:
            with st.spinner("Summarizing text..."):
                summary = summarize_text(text)
            st.subheader("Summary")
            st.write(summary)

if __name__ == "__main__":
    main()