# PDF to DOC Converter with AI and Multi-Class Prediction

This application converts PDF files to DOC format using AI enhancement and automatically classifies the document type using machine learning.

## Features

- Extract text from PDF files
- Enhance text formatting and readability using AI
- Convert to DOC format with proper formatting
- Automatic document type classification
- Page-by-page processing

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Replace "your-api-key-here" in `pdf_to_doc.py` with your OpenAI API key
2. Place your PDF file in the project directory
3. Run the script:
```bash
python pdf_to_doc.py
```

The program will:
1. Convert your PDF to DOC
2. Use AI to enhance the text
3. Classify the document type
4. Save the output as `output.docx`

## Requirements

- Python 3.8+
- OpenAI API key
- PyMuPDF
- python-docx
- openai
- scikit-learn
- numpy

## Example

The project includes a `test.pdf` file that you can use to test the system. When run, it will:
1. Convert test.pdf to output.docx
2. Classify the document type
3. Show the classified type in the console
