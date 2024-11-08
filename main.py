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
import logging
import sys
from tqdm import tqdm

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

def summarize_content(text, model, context_window=8192):
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

Begin the summary below:"""
    
    prompt = f"{text}"
    response = completion(model=model, messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ])
    return response.choices[0].message.content

def analyze_summaries(summaries_dict, model):
    """Synthesize summaries into a cohesive and detailed report."""
    system_prompt = """You are an expert analyst skilled in synthesizing information from multiple documents to create a cohesive and detailed report.

**Your tasks are:**
- Integrate the provided summaries into a unified report.
- Highlight connections, relationships, and any contradictions between the documents.
- Provide a comprehensive overview that captures the collective insights of all summaries.
- Present the analysis in an organized manner with clear headings or sections if necessary.
- Identify any gaps or missing information that could be relevant.

**Guidelines:**
- Use objective language and maintain neutrality.
- Support your analysis with evidence from the summaries.
- Ensure the synthesis is accessible to readers without prior knowledge of the documents.
- Use bullet points or numbered lists where appropriate to enhance clarity.

Begin your synthesis below:"""
    
    formatted_summaries = "\n\n".join([f"Document '{k}':\n{v}" for k, v in summaries_dict.items()])
    user_prompt = f"{formatted_summaries}"
        
    response = completion(model=model, messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])
    return response.choices[0].message.content

def produce_response(user_query, synthesis, model):
    """Generate final response based on synthesis and user query."""
    system_prompt = """You are a knowledgeable assistant proficient in providing detailed and accurate answers to user queries based on synthesized information from multiple sources.

**Your tasks are:**
- Carefully read the user's query and understand their information needs.
- Utilize the synthesized document information to construct your response.
- Provide clear, precise, and well-structured answers that directly address the query.
- Include relevant details, examples, or explanations from the synthesis to support your answer.
- Ensure your response is accurate, unbiased, and helpful.

**Guidelines:**
- Maintain a formal and informative tone.
- Do not introduce information not present in the synthesis.
- If the synthesis lacks information to answer the query fully, acknowledge this and provide the best possible answer based on available data.

Begin your response below:"""
    
    user_prompt = f"""Synthesized Information:
{synthesis}

User Query:
{user_query}"""
    
    response = completion(model=model, messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])
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
        return file_name, summary
    except Exception as e:
        logger.error(f"Error processing {file_name}: {str(e)}")
        return file_name, ""

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
    logger.info("Starting document processing pipeline")
    global global_model
    global_model = model

    logger.info(f"Scanning directory: {directory}")
    files = [(os.path.join(directory, f), f) for f in os.listdir(directory)]
    logger.info(f"Found {len(files)} files to process")

    logger.info("Processing files in parallel")
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(process_file_wrapper, files),
            total=len(files),
            desc="Processing files"
        ))

    summaries_dict = dict(results)
    logger.info(f"Generated summaries for {len(summaries_dict)} files")

    logger.info("Analyzing summaries")
    synthesis = analyze_summaries(summaries_dict, model)
    # print(synthesis)

    logger.info("Generating final response")
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
    parser.add_argument('-v', type=int, choices=[0, 1, 2, 3], default=0, help='Verbosity level: 0=INFO, 1=DEBUG, 2=WARNING, 3=ERROR')

    args = parser.parse_args()

    global logger
    logger = setup_logging(args.v)

    logger.info(f"Starting processing with model: {args.m}")
    logger.info(f"Source directory: {args.s}")
    
    try:
        model = f"ollama_chat/{args.m}"
        llm_response = process_documents(args.s, model, args.q)

        print("\nResponse:")
        print(llm_response)

        if args.o:
            with open(args.o, 'w', encoding='utf-8') as f:
                f.write(llm_response)
            logger.info(f"Response saved to {args.o}")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()