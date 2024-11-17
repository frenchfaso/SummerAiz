# SummerAiz

Welcome to SummerAiz, your local document summarization system designed to make sense of all the docs in a folder. Whether it's PDFs, DOCXs, TXTs, MDs, or even those pesky EMLs, SummerAiz has got you covered.

## What Does It Do?

SummerAiz scans your specified directory, processes each document, and extracts the text. It then uses a language model to summarize the content and generate a cohesive report for each document.

## Why Use SummerAiz?

- **Automated Document Processing**: No more manual reading and summarizing. Let SummerAiz do the heavy lifting.
- **Multi-format Support**: Handles various document formats like a pro.
- **Smart Summarization**: Summarizes content with precision and clarity.
- **Accurate Summaries**: Generates well-structured summaries for your documents.
- **Improved Privacy**: Utilizes local LLMs (ollama required) to ensure your data stays private.

## How to Use It?

1. **Install Dependencies**: Make sure you have all the required Python packages installed.
2. **Install Ollama**: Follow the instructions [here](https://ollama.com) to install Ollama.
3. **Run the Script**: Use the command line to run the script with your source folder.

### Command-line Options

- `-m`: Model name, defaults to "gemma2".
- `-s`: Source folder (where the system will search for document files) (required).
- `-o`: Optional, a filename to save the output as a .md markdown file (the summaries).
- `-v`: Verbosity level: 0=INFO, 1=DEBUG, 2=WARNING, 3=ERROR (default is 0).

### Example Usage
```sh
python main.py -s /path/to/source/folder -o output.md
```