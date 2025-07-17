-----

# ask\_pdf\_ai

Ask\_PDF\_AI is a powerful and intelligent assistant that allows you to chat with your PDF documents. Powered by Google's **Gemini AI** and **ChromaDB**, this tool extracts information from your PDFs, creates semantic chunks, and provides accurate answers to your questions based on the document's content.

-----

## Features

  * **PDF Text Extraction**: Seamlessly extracts text content from PDF files.
  * **Semantic Chunking**: Intelligently splits the extracted text into meaningful semantic chunks (paragraphs) for better understanding and retrieval.
  * **Gemini Embeddings**: Utilizes the `embedding-001` Gemini model to generate high-quality vector embeddings for text chunks.
  * **ChromaDB Integration**: Stores and manages the embedded text chunks efficiently using ChromaDB, a lightweight and fast vector database.
  * **Retrieval-Augmented Generation (RAG)**: Combines document retrieval with Gemini's generative capabilities (`gemini-2.5-flash`) to provide contextually relevant and accurate answers.
  * **Persistent Storage**: ChromaDB stores its data locally, so your indexed PDFs and embeddings are saved for future sessions.
  * **Comprehensive Logging**: Detailed logging helps in monitoring the application's flow and debugging.

-----

## Getting Started

Follow these steps to set up and run the Ask\_PDF\_AI assistant.

### Prerequisites

  * Python 3.9+
  * Google Gemini API Key

### Installation

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/RishinR/ask_pdf_ai.git
    cd ask_pdf_ai
    ```

2.  **Create a virtual environment** (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your Google Gemini API Key**:

    Create a `.env` file in the root directory of the project and add your Gemini API key:

    ```
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    ```

### Usage

1.  **Place your PDF file**:

    Put your PDF document inside the `files/` directory. By default, the script looks for `files/rust_book.pdf`. You can change the `PDF_FILE_PATH` variable in the script to point to your desired PDF.

    ```python
    # --- Configuration ---
    PDF_FILE_PATH = "files/your_document.pdf"  # Update this line
    ```

2.  **Run the script**:

    ```bash
    python main.py
    ```

    The first time you run the script, it will:

      * Extract text from your PDF.
      * Create semantic chunks.
      * Generate embeddings using the Gemini `embedding-001` model.
      * Store these embeddings and document chunks in a local ChromaDB instance located in the `./chroma_db` directory.

    Subsequent runs will load the existing ChromaDB collection, making the startup faster.

3.  **Start asking questions**:

    Once the initialization is complete, you'll be prompted to ask questions about your PDF:

    ```
    --- ASK_PDF_AI Assistant ---
    Type your questions about the resume. Type 'exit' to stop.

    Your question: What is Rust programming language known for?
    ```

    Type your question and press Enter. The AI assistant will retrieve relevant information from your PDF and generate an answer using the `gemini-2.5-flash` model.

    To exit the assistant, type `exit` and press Enter.

-----

## Project Structure

```
ask_pdf_ai/
├── files/
│   └── your_document.pdf  # Your PDF documents go here
├── .env               # Environment variables (e.g., GEMINI_API_KEY)
├── maiN.py # Main script (e.g., main.py or ask_pdf.py)
├── requirements.txt   # Python dependencies
└── chroma_db/         # Directory for ChromaDB persistence (created automatically)
    └── ...            # ChromaDB files
```

-----

## Configuration

You can modify the following variables in the script:

  * `PDF_FILE_PATH`: Path to your PDF file. Default: `"files/your_document.pdf"`.
  * `CHROMA_DB_PATH`: Directory where ChromaDB data will be stored. Default: `"./chroma_db"`.
  * `COLLECTION_NAME`: The name of the collection in ChromaDB. Default: `"pdf_chunks"`.
  * `embedding_model`: The Gemini embedding model to use. Default: `"models/embedding-001"`.
  * `GENERATIVE_MODEL`: The Gemini generative model to use for answers. Default: `"gemini-2.5-flash"`.
  * `LOG_FILE`: Path for the application's log file. Default: `"rag_system.log"`.

-----

## Logging

The system logs various operations, warnings, and errors to `rag_system.log`. This file can be helpful for debugging and understanding the assistant's behavior.

-----

## Contributing

Feel free to fork the repository, make improvements, and submit pull requests.

-----

## License

This project is open-source and available under the [MIT License](https://www.google.com/search?q=LICENSE).

-----