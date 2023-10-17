from fastapi import FastAPI, File, UploadFile
from pyngrok import ngrok
from typing import List
import uvicorn
import nest_asyncio

import easyocr

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import numpy as np
import requests
from pdf2image import convert_from_bytes
import io

app = FastAPI()

@app.post("/QA")
def load_file(file_url: str, sentences: List[str]):

        # Download the PDF from the URL
        response = requests.get(file_url)
        response.raise_for_status()

        # Convert the PDF content to images
        pdf_bytes = response.content
        pages = convert_from_bytes(pdf_bytes)

        # Initialize the OCR reader
        reader = easyocr.Reader(["en"], gpu=True)

        # Initialize an empty variable to store the extracted text
        all_text = ''

        # Loop through each page and perform OCR
        for page_num, page in enumerate(pages, start=1):
            print(f"Processing page {page_num}...")
            # Convert PIL image to numpy array
            page_np = np.array(page)

            text = reader.readtext(page_np)
            for result in text:
                bbox, text, prob = result
                all_text += f"{text} "
            all_text += '\n'


        model_name = "deepset/roberta-base-squad2"

        nlp = pipeline('question-answering', model=model_name, tokenizer=model_name,device=0)

        # Define the common context
        context = all_text

        # List of questions
        questions = sentences
        # Initialize an empty dictionary to store questions and answers
        qa_dict = {}
        # Get answers for each question with the same context
        for question in questions:
            QA_input = {
                'question': question,
                'context': context
            }
            res = nlp(QA_input)
            print(f"Question: {question}")
            print(f"Answer: {res['answer']}")
            qa_dict[question] = res['answer']


        return qa_dict
