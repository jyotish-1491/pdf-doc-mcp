import os
import fitz  # PyMuPDF
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
import openai
import json
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import numpy as np

class PDFToDocConverter:
    def __init__(self, api_key: str):
        self.api_key = api_key
        openai.api_key = api_key
        self.classifier = self._create_classifier()
        self.label_encoder = LabelEncoder()
        
    def _create_classifier(self) -> Pipeline:
        """Create and return the multi-class classifier."""
        return Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('classifier', MultinomialNB())
        ])
        
    def train_classifier(self, training_data: List[Tuple[str, str]]):
        """Train the classifier with labeled data."""
        texts, labels = zip(*training_data)
        labels_encoded = self.label_encoder.fit_transform(labels)
        self.classifier.fit(texts, labels_encoded)
        
    def predict_document_type(self, text: str) -> str:
        """Predict the type of document based on its content."""
        prediction = self.classifier.predict([text])
        return self.label_encoder.inverse_transform(prediction)[0]
        
    def extract_text_from_pdf(self, pdf_path: str) -> List[str]:
        """Extract text from PDF pages."""
        text_pages = []
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text = page.get_text("text")
                    text_pages.append(text)
            return text_pages
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def enhance_text_with_ai(self, text: str) -> str:
        """Enhance text using OpenAI's API."""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a text enhancement assistant. Improve the formatting and readability of the text."},
                    {"role": "user", "content": text}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error enhancing text with AI: {str(e)}")

    def create_docx(self, text_pages: List[str], output_path: str):
        """Create DOCX file from text pages."""
        doc = Document()
        
        for page in text_pages:
            enhanced_text = self.enhance_text_with_ai(page)
            
            # Add page content
            paragraph = doc.add_paragraph(enhanced_text)
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            
            # Add page break except for last page
            if page != text_pages[-1]:
                doc.add_page_break()
        
        try:
            doc.save(output_path)
        except Exception as e:
            raise Exception(f"Error saving DOCX file: {str(e)}")

    def convert_pdf_to_doc(self, pdf_path: str, output_path: str):
        """Main conversion function."""
        try:
            # Extract text from PDF
            text_pages = self.extract_text_from_pdf(pdf_path)
            
            # Create DOCX file
            self.create_docx(text_pages, output_path)
            
            return True
        except Exception as e:
            raise Exception(f"Conversion failed: {str(e)}")

def main():
    # Example usage
    converter = PDFToDocConverter(api_key="your-api-key-here")
    
    # Example training data (in real use, this would be much larger)
    training_data = [
        ("This is a report about quarterly sales", "Report"),
        ("Dear Sir/Madam, I am writing to complain...", "Complaint"),
        ("To whom it may concern, I am applying for the position...", "Application"),
        ("The purpose of this manual is to provide...", "Manual")
    ]
    
    # Train the classifier
    converter.train_classifier(training_data)
    
    # Example conversion
    try:
        output_path, doc_type = converter.convert_pdf_to_doc(
            "example.pdf",
            "output.docx"
        )
        print(f"Conversion successful! Document type: {doc_type}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
