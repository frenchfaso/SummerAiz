import os
import docx
import pdfplumber
from email import policy
from email.parser import BytesParser
import argparse
from litellm import completion
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

global_model = None

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
    system_prompt = "You are a summarization expert. Your task is to provide concise and accurate summaries of given texts."
    prompt = f"Please provide a concise summary of the following text:\n\n{text}"
    response = completion(model=model, messages=[
        {"content": system_prompt, "role": "system"},
        {"content": prompt, "role": "user"}
    ])
    return response.choices[0].message.content

def analyze_summaries(summaries_dict, model):
    """Synthesize summaries into a coherent narrative."""
    system_prompt = """You are an expert analyst specializing in synthesizing information from multiple sources.
Your task is to create a cohesive narrative that:
- Identifies common themes across documents
- Highlights key connections and contradictions
- Provides a comprehensive overview of the collected information
Be concise but thorough in your analysis."""

    formatted_summaries = "\n\n".join([f"Document '{k}':\n{v}" for k, v in summaries_dict.items()])
    user_prompt = f"Please analyze and synthesize these document summaries:\n\n{formatted_summaries}"
    
    response = completion(model=model, messages=[
        {"content": system_prompt, "role": "system"},
        {"content": user_prompt, "role": "user"}
    ])
    return response.choices[0].message.content

def produce_response(user_query, synthesis, model):
    """Generate final response based on synthesis and user query."""
    system_prompt = """You are an expert information producer specializing in answering queries based on synthesized content.
Your task is to:
- Generate precise and relevant responses to user queries
- Draw from the provided document synthesis
- Present information in a clear and coherent manner
Focus on accuracy and relevance in your responses."""

    user_prompt = f"""Using this document synthesis as context:

{synthesis}

Query: {user_query}"""

    response = completion(model=model, messages=[
        {"content": system_prompt, "role": "system"},
        {"content": user_prompt, "role": "user"}
    ])
    return response.choices[0].message.content

def process_file(file_data, model):
    file_path, file_name = file_data
    # Extract text using existing methods
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
        return file_name, ""
    
    # Summarize the extracted text
    summary = summarize_content(text, model)
    return file_name, summary

def process_file_wrapper(file_data):
    return process_file(file_data, global_model)

def build_file_text_dict(directory_path):
    file_text_dict = {}
    files_to_process = []
    
    # Collect all files first
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            files_to_process.append((file_path, file))
    
    # Process files in parallel using all available cores
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results = executor.map(process_file, files_to_process)
        
        for file_name, content in results:
            if content:  # Only add if content was successfully extracted
                file_text_dict[file_name] = content
                
    return file_text_dict


def process_documents(directory, model, user_query):
    # Set the global model variable so it can be accessed by worker processes
    global global_model
    global_model = model

    # Get list of files
    files = [(os.path.join(directory, f), f) for f in os.listdir(directory)]

    # Process files in parallel with summarization
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_file_wrapper, files))

    # Create dictionary of summaries
    summaries_dict = dict(results)
    # print(summaries_dict)

    # Generate synthesis of summaries
    synthesis = analyze_summaries(summaries_dict, model)
    print(synthesis)

    # Produce final response
    final_response = produce_response(user_query, synthesis, model)

    return final_response

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

    # Process documents through multi-agent pipeline
    model = f"ollama_chat/{args.m}"
    llm_response = process_documents(args.s, model, args.q)

    # Print response to terminal
    print("\nResponse:")
    print(llm_response)

    if args.o:
        with open(args.o, 'w', encoding='utf-8') as f:
            f.write(llm_response)
        print(f"\nResponse saved to {args.o}")

if __name__ == "__main__":
    main()