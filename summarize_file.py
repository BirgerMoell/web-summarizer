import streamlit as st
import pdfplumber
from PIL import Image
import pytesseract
#from transformers import pipeline
import io

import os
from dotenv import load_dotenv

# groq
from groq import Groq

# SwedishBeagle-dare
from transformers import AutoTokenizer
import transformers
import torch

class Summarizer:

    def __init__(self, model = "groq"):
        self.model = model
        self.client = self.load_groq()

    def run_app(self):
        st.title("Text Extractor and Summarizer")
    
        uploaded_file = st.file_uploader("Upload an Image or PDF", type=["jpg", "jpeg", "png", "pdf"])
    
        if uploaded_file is not None:
            if uploaded_file.type == "application/pdf":
                with st.spinner("Extracting text from PDF..."):
                    text = self.extract_text_from_pdf(uploaded_file)
            else:
                image = Image.open(uploaded_file)
                with st.spinner("Extracting text from image..."):
                    text = self.extract_text_from_image(image)
    
            if text:
                with st.spinner("Summarizing text..."):
                    summary = self.summarize_using_groq(text)
                st.subheader("Summary")
                st.write(summary)
    
            st.subheader("Extracted Text")
            st.write(text)


    # Function to extract text from an image
    def extract_text_from_image(self, image):
        text = pytesseract.image_to_string(image)
        return text
    
    # Function to extract text from a PDF
    def extract_text_from_pdf(self, pdf):
        text = ""
        with pdfplumber.open(pdf) as pdf_file:
            for page in pdf_file.pages:
                text += page.extract_text()
        return text
    
    # Function to summarize text
    #def summarize_text(self, text):
    #    summarizer = pipeline("summarization")
    #    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    #    return summary[0]['summary_text']
    
    def load_groq(self):
        load_dotenv()
    
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
        client = Groq(
            api_key=GROQ_API_KEY
        )
    
        return client

    def summarize_using_groq(self, text):
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You summarize texts that the users sends"
                },
                {
                    "role": "user",
                    "content": text,
                }
            ],
            model="mixtral-8x7b-32768",
        )
    
        return chat_completion.choices[0].message.content
    
    def summarize_using_swedishbeagle(self, text):
        # https://huggingface.co/FredrikBL/SwedishBeagle-dare

        model = "FredrikBL/SwedishBeagle-dare"
        messages = [
            {
                "role": "system", 
                "content": "You summarize texts that the users sends"
            }, 
            {
                "role": "user", 
                "content": text
            }
        ]

        tokenizer = AutoTokenizer.from_pretrained(model)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        outputs = pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        return outputs[0]["generated_text"]

    def summarize(self, text):
        if(self.model == "groq"):
            return self.summarize_using_groq(text)
        elif(self.model == "SwedishBeagle-dare"):
            return self.summarize_using_swedishbeagle(text)
# Streamlit app
def main():
    # Models:
    # - groq
    # - SwedishBeagle-dare
    summarizer = Summarizer(model="SwedishBeagle-dare")
    summarizer.run_app()

if __name__ == "__main__":
    main()