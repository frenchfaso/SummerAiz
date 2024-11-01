import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from crewai import Agent, Task, Crew
from langchain.tools import Tool
from langchain_community.llms import Ollama

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    SUPPORTED_EXTENSIONS = {'.txt', '.pdf', '.docx', '.md', '.eml'}
    
    @staticmethod
    def get_documents(folder_path: str) -> List[Path]:
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Folder not found: {folder_path}")
        
        documents = []
        for file in folder.rglob('*'):
            if file.suffix.lower() in DocumentProcessor.SUPPORTED_EXTENSIONS:
                documents.append(file)
            else:
                logger.warning(f"Skipping unsupported file: {file}")
        return documents

def create_agents(model_name: str) -> Dict[str, Agent]:
    llm = Ollama(model=model_name)
    
    summarizer = Agent(
        name="Summarizer",
        llm=llm,
        role="Document Summarizer",
        goal="Create concise summaries of documents while preserving key information",
        tools=[Tool(
            name="read_file",
            func=lambda x: open(x).read(),
            description="Read content of a file"
        )]
    )
    
    analyzer = Agent(
        name="Analyzer",
        llm=llm,
        role="Content Analyzer",
        goal="Synthesize document summaries into a cohesive narrative"
    )
    
    producer = Agent(
        name="Producer",
        llm=llm,
        role="Content Producer",
        goal="Create final response addressing the user's query based on analyzed content"
    )
    
    return {"summarizer": summarizer, "analyzer": analyzer, "producer": producer}

def create_tasks(agents: Dict[str, Agent], documents: List[Path], query: str) -> List[Task]:
    tasks = []
    
    # Create summarization tasks for each document
    for doc in documents:
        tasks.append(Task(
            description=f"Summarize the content of {doc}",
            agent=agents["summarizer"]
        ))
    
    # Create analysis task
    tasks.append(Task(
        description="Synthesize all document summaries into a cohesive narrative",
        agent=agents["analyzer"]
    ))
    
    # Create production task
    tasks.append(Task(
        description=f"Create final response addressing the query: {query}",
        agent=agents["producer"]
    ))
    
    return tasks

def save_output(content: str, output_path: str, query: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    markdown_content = f"""# Document Analysis Results

## Analysis
{content}

---
**Query:** {query}
**Generated:** {timestamp}
"""
    
    with open(output_path, 'w') as f:
        f.write(markdown_content)

def main():
    parser = argparse.ArgumentParser(description='Document Analysis Tool')
    parser.add_argument('-m', '--model', required=True, help='Local Ollama model name')
    parser.add_argument('-q', '--query', required=True, help='Main user query')
    parser.add_argument('-f', '--folder', required=True, help='Target folder path')
    parser.add_argument('-o', '--output', help='Output file path (optional)')
    
    args = parser.parse_args()
    
    try:
        # Get list of documents
        documents = DocumentProcessor.get_documents(args.folder)
        if not documents:
            raise ValueError(f"No supported documents found in {args.folder}")
        
        # Create agents and tasks
        agents = create_agents(args.model)
        tasks = create_tasks(agents, documents, args.query)
        
        # Create and run crew
        crew = Crew(
            agents=list(agents.values()),
            tasks=tasks
        )
        result = crew.kickoff()
        
        # Handle output
        if args.output:
            save_output(result, args.output, args.query)
            logger.info(f"Results saved to {args.output}")
        else:
            print("\nAnalysis Results:")
            print("----------------")
            print(result)
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()