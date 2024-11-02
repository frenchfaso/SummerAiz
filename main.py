import os
import docx
import pdfplumber
from email import policy
from email.parser import BytesParser
import argparse
import ollama

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

def build_file_text_dict(directory_path):
    file_text_dict = {}
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.docx'):
                file_text_dict[file] = extract_text_from_docx(file_path)
            elif file.endswith('.pdf'):
                file_text_dict[file] = extract_text_from_pdf(file_path)
            elif file.endswith('.txt'):
                file_text_dict[file] = extract_text_from_txt(file_path)
            elif file.endswith('.md'):
                file_text_dict[file] = extract_text_from_md(file_path)
            elif file.endswith('.eml'):
                file_text_dict[file] = extract_text_from_eml(file_path)
    return file_text_dict

def main():
    parser = argparse.ArgumentParser(
        description='Process some documents and generate responses using a language model.',
        epilog='Example usage:\n'
               '  python main.py -q "What is the capital of France?" -s /path/to/source/folder\n'
               '  python main.py -q "What is the capital of France?" -s /path/to/source/folder -o output.md',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-m', default='gemma2', help='Model name, defaults to "gemma2"')
    parser.add_argument('-q', required=True, help='User prompt (the question the agentic system will answer)')
    parser.add_argument('-s', required=True, help='Source folder (where the system will search for document files)')
    parser.add_argument('-o', help='Optional, a filename to save the output as a .md markdown file (the llm response)')

    args = parser.parse_args()

    # Example usage:
    directory_path = args.s
    # file_texts = build_file_text_dict(directory_path)

    # Placeholder for LLM response
    response = ollama.chat(model='gemma2', messages=[
        {
            'role': 'user',
            'content': args.q,
        },
    ])
    llm_response = response['message']['content']

    if args.o:
        with open(args.o, 'w', encoding='utf-8') as output_file:
            output_file.write(llm_response)
    else:
        print(llm_response)

if __name__ == '__main__':
    main()