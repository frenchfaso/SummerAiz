#!/usr/bin/env python

import os
import docx
import pdfplumber
from email import policy
from email.parser import BytesParser
import argparse
from litellm import completion
import logging
import sys

def setup_logging(verbosity):
    level = logging.INFO
    if verbosity == 1:
        level = logging.DEBUG
    elif verbosity == 2:
        level = logging.WARNING
    elif verbosity == 3:
        level = logging.ERROR

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
        
    return logging.getLogger(__name__)

logger = None

global_model = None
global_context_window = 8192

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return '\n'.join([para.text for para in doc.paragraphs])

def extract_text_from_pdf(file_path):
    text = ''
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + '\n'
    return text

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text_from_md(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text_from_eml(file_path):
    with open(file_path, 'rb') as file:
        msg = BytesParser(policy=policy.default).parse(file)
        return msg.get_body(preferencelist=('plain')).get_content()

def summarize_content(text, model):
    """Summarize document content using the specified model."""
    system_prompt = """You are an expert summarization assistant specializing in analyzing various types of documents.

**Your tasks are:**
- Carefully read and comprehend the provided text.
- Identify key points, main arguments, significant details, and conclusions.
- Extract relevant information and produce a detailed summary of the document.
- Ensure the summary is concise (approximately 200-250 words), coherent, and free of personal opinions or biases.

**Guidelines:**
- Use clear and direct language.
- Avoid technical jargon unless necessary, and explain any essential terms.
- Do not include unnecessary details or repeat information.
- Maintain the original context and meaning of the document.
- Highlight any critical insights or findings.

**Formatting:**
- Start with a brief introduction if necessary.
- Use bullet points for lists or key points.
- Ensure proper grammar and punctuation.

Begin the summary below:"""
    
    prompt = f"{text}"
    response = completion(model=model, messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ],
    num_ctx=global_context_window)
    return response.choices[0].message.content

def process_file(file_data, model):
    file_path, file_name = file_data
    logger.debug(f"Processing file: {file_name}")
    try:
        if file_name.endswith('.docx'):
            text = extract_text_from_docx(file_path)
        elif file_name.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif file_name.endswith('.txt'):
            text = extract_text_from_txt(file_path)
        elif file_name.endswith('.md'):
            text = extract_text_from_md(file_path)
        elif file_name.endswith('.eml'):
            text = extract_text_from_eml(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_name}")
            return file_name, ""
        
        summary = summarize_content(text, model)
        print(f"File: '{file_name}'\nSummary: '{summary}'\n")
        return file_name, summary
    except Exception as e:
        logger.error(f"Error processing {file_name}: {str(e)}")
        return file_name, ""

def process_documents(directory, model):
    logger.info("Starting document processing pipeline")
    global global_model
    global_model = model

    logger.info(f"Scanning directory: {directory}")
    files = [(os.path.join(directory, f), f) for f in os.listdir(directory)]
    logger.info(f"Found {len(files)} files to process")

    for file_data in files:
        process_file(file_data, model)

def main():
    parser = argparse.ArgumentParser(
        description='Process some documents and generate summaries using a language model.',
        epilog='Example usage:\n'
               '  python main.py -s /path/to/source/folder\n'
               '  python main.py -s /path/to/source/folder -o output.md',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-m', default='gemma2', help='Model name, defaults to "gemma2"')
    parser.add_argument('-s', required=True, help='Source folder (where the system will search for document files)')
    parser.add_argument('-o', help='Optional, a filename to save the output as a .md markdown file (the summaries)')
    parser.add_argument('-v', type=int, choices=[0, 1, 2, 3], default=0, help='Verbosity level: 0=INFO, 1=DEBUG, 2=WARNING, 3=ERROR')

    args = parser.parse_args()

    global logger
    logger = setup_logging(args.v)

    logger.info(f"Starting processing with model: {args.m}")
    logger.info(f"Source directory: {args.s}")
    
    try:
        model = f"ollama_chat/{args.m}"
        process_documents(args.s, model)

        if args.o:
            with open(args.o, 'w', encoding='utf-8') as f:
                for file_data in [(os.path.join(args.s, f), f) for f in os.listdir(args.s)]:
                    file_name, summary = process_file(file_data, model)
                    if summary:
                        f.write(f"File: '{file_name}'\nSummary: '{summary}'\n\n")
            logger.info(f"Summaries saved to {args.o}")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()