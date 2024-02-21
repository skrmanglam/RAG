# GenAI in Retrieval Augmented Generation (RAG) Systems

This repository houses an innovative expert system leveraging Generative AI (GenAI) and Language Models (LLMs) in Retrieval Augmented Generation (RAG) frameworks. Designed to digest information from PDF documents and web URLs, this system dynamically learns, stores knowledge embeddings, and utilizes LLMs for intelligent information retrieval and generation.

## Overview

The system comprises three primary components:

1. **RAG_pdf**: Handles information retrieval from PDF documents. (Note: Currently utilizes an older version of Langchain and requires updates.)
2. **RAG_web**: Specializes in information retrieval from web URLs.
3. **Expert_rag**: Integrates both PDF and web retrieval capabilities, allowing users to query a consolidated knowledge base. Current limitations include session-based memory and singular data source processing per session.

## Getting Started

Follow these steps to run the system locally:

1. **Clone the Repository**: Clone this repo to your local machine.
2. **Python Installation**: Ensure Python 3.10 or higher is installed on your system.
3. **Dependency Installation**: Install all required dependencies listed in `requirements.txt`. It's recommended to use a virtual environment for this purpose.
4. **API Tokens**: Obtain necessary API tokens from Hugging Face.
5. **Launch the App**: Run the application using `streamlit run app.py`. Navigate to the localhost URL in your browser to interact with the app.

### Installation Guide

```bash
git clone <repo-url>
cd <repo-directory>
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
# Set up your environment variables or .env file with the Hugging Face API token
streamlit run app.py
```

### References:
The initial codebase was inspired by [chat-with-websites](https://github.com/alejandro-ao/chat-with-websites), adapted for open-source use and enhanced to support dual data sources without relying on proprietary models like OpenAI's embeddings.

### Planned Enhancements
Input Flexibility: Enable seamless switching between URL and PDF inputs without needing to reset the app.
Persistent Knowledge Base: Integrate long-term database storage for accumulated knowledge.
Local and Secure LLM Usage: Transition to GPT4ALL for local, secure LLM execution.
Performance Optimization: Implement speed enhancements for improved user experience.

### Contributing
Contributions to improve the system are welcome. Whether it's feature enhancements, bug fixes, or documentation improvements, please feel free to fork the repo and submit a pull request.
