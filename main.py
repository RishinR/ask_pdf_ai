import os
import pypdf
import chromadb
from dotenv import load_dotenv
import google.generativeai as genai
# Import EmbeddingFunction from the correct ChromaDB types module
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
import logging

# Load environment variables
load_dotenv()

# --- Configuration ---
PDF_FILE_PATH = "files/rust_book.pdf"  # Replace with your PDF file path
CHROMA_DB_PATH = "./chroma_db"     # Directory to store ChromaDB data
COLLECTION_NAME = "pdf_chunks"

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
embedding_model = "models/embedding-001" # Gemini embedding model
GENERATIVE_MODEL = "gemini-2.5-flash" # Gemini model for generating answers

# --- Logging Configuration ---
LOG_FILE = "rag_system.log"

# Create a logger instance
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Set the logger's overall level to DEBUG

# Create a file handler
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.DEBUG) # Set the file handler's level to DEBUG

# Create a formatter and add it to the file handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)

# --- Custom Gemini Embedding Function ---
# Define the class outside the function for better scope and reusability
class GeminiEmbeddingFunction(EmbeddingFunction[Documents]): # Inherit from chromadb.api.types.EmbeddingFunction
    def __init__(self, model_name: str):
        # We don't need to call super().__init__() here if EmbeddingFunction
        # is a Protocol or doesn't have an __init__ that needs explicit calling.
        # The warning implies the *lack* of an __init__ in our class was the issue.
        self.model_name = model_name

    def __call__(self, texts: Documents) -> Embeddings:
        embeddings_list = []
        for i, text in enumerate(texts):
            try:
                response = genai.embed_content(model=self.model_name, content=text)
                embeddings_list.append(response['embedding'])
            except Exception as e:
                logger.error(f"Error embedding chunk {i} with Gemini: {e}")
                # Fallback to a zero vector of the correct dimension
                # Fetching model info to get dynamic embedding dimension
                try:
                    dim = genai.get_model(self.model_name).embedding_dimension
                except Exception:
                    dim = 768 # Default if model info isn't available
                embeddings_list.append([0.0] * dim)
        return embeddings_list

# --- 1. PDF to Text Extraction ---
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file, page by page.
    Returns a list of strings, where each string is the text of a page.
    """
    text_content = []
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return []

    try:
        reader = pypdf.PdfReader(pdf_path)
        for page in reader.pages:
            text_content.append(page.extract_text() or "")
        logger.info(f"Successfully extracted text from {len(text_content)} pages from {os.path.basename(pdf_path)}.")
    except Exception as e:
        logger.exception(f"Error extracting text from PDF '{pdf_path}': {e}")
    return text_content

# --- 2. Semantic Chunking (Paragraph-level for simplicity) ---
def create_semantic_chunks(text_pages):
    """
    Splits the text from PDF pages into semantic chunks (paragraphs).
    Filters out empty chunks.
    """
    chunks = []
    if not text_pages:
        logger.warning("No text pages provided for chunking.")
        return []

    for page_num, page_text in enumerate(text_pages):
        # Split by double newline to get paragraphs
        paragraphs = [p.strip() for p in page_text.split('\n\n') if p.strip()]
        for para_num, paragraph in enumerate(paragraphs):
            # Add some metadata for easier tracing
            chunks.append({
                "content": paragraph,
                "metadata": {"page": page_num + 1, "paragraph": para_num + 1}
            })
    logger.info(f"Created {len(chunks)} semantic chunks.")
    return chunks

# --- 3. Embed and Store in ChromaDB ---
def embed_and_store_chunks(chunks, db_path, collection_name, embedding_model_name):
    """
    Generates embeddings for chunks using Gemini and stores them in ChromaDB.
    """
    client = chromadb.PersistentClient(path=db_path)

    # Instantiate the custom embedding function here
    gemini_ef = GeminiEmbeddingFunction(model_name=embedding_model_name)

    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=gemini_ef # Use the custom Gemini embedding function
        )
        logger.info(f"ChromaDB collection '{collection_name}' initialized/retrieved.")

        # Check if the collection is empty. If so, add documents.
        if collection.count() == 0:
            if not chunks:
                logger.warning("No chunks provided to add to ChromaDB.")
                return collection

            documents_to_add = [chunk["content"] for chunk in chunks]
            metadatas_to_add = [chunk["metadata"] for chunk in chunks]
            ids_to_add = [f"chunk_{i}" for i in range(len(chunks))]

            collection.add(
                documents=documents_to_add,
                metadatas=metadatas_to_add,
                ids=ids_to_add
            )
            logger.info(f"Successfully added {len(documents_to_add)} chunks to ChromaDB.")
        else:
            logger.info(f"ChromaDB collection '{collection_name}' already contains {collection.count()} items. Skipping embedding and storage.")
        return collection
    except Exception as e:
        logger.exception(f"Error interacting with ChromaDB during embedding/storage: {e}")
        return None

# --- 4. Query ChromaDB ---
def query_chroma_db(collection, query_text, n_results=5, embedding_model_name=embedding_model):
    """
    Queries ChromaDB with a given text and retrieves top results.
    """
    if not collection:
        logger.error("ChromaDB collection not available for querying.")
        return []

    try:
        # Embed the query text using the same Gemini embedding model
        # Use the base genai.embed_content for query embedding as it's a one-off call
        query_embedding_response = genai.embed_content(model=embedding_model_name, content=query_text)
        query_embedding = query_embedding_response['embedding']

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        logger.info(f"Retrieved {len(results['documents'][0])} relevant chunks for query: '{query_text[:50]}...'")
        return results
    except Exception as e:
        logger.exception(f"Error querying ChromaDB with query '{query_text[:50]}...': {e}")
        return []

# --- 5. Generate Answer using Gemini (RAG) ---
def generate_gemini_response(query, retrieved_chunks_content):
    """
    Uses Gemini to generate a coherent answer based on the query and retrieved content.
    """
    model = genai.GenerativeModel(GENERATIVE_MODEL)

    if not retrieved_chunks_content:
        logger.warning("No retrieved chunks provided for Gemini response generation.")
        return "I apologize, but I couldn't find relevant information to answer your question from the document."

    context = "\n\n".join(retrieved_chunks_content)

    prompt = (
        f"""
            You are a helpful assistant specialized in answering questions based on provided documents.
            Here is the context extracted from the document:
            {context}
            Based on this context, please answer the following question:
            {query}
            Provide a concise and accurate answer, ensuring it is grounded in the provided context.
            If the answer is not available in the provided context, state that clearly and politely.
            Make the conversation natural and informative.
        """
    )

    try:
        response = model.generate_content(prompt)
        logger.info("Successfully generated Gemini response.")
        return response.text
    except Exception as e:
        logger.exception(f"Error generating Gemini response for query '{query[:50]}...': {e}")
        return "I apologize, but I couldn't generate an answer based on the provided information due to an internal error."

# --- Main Execution ---
if __name__ == "__main__":
    # Ensure the 'files' directory exists for the PDF
    if not os.path.exists(os.path.dirname(PDF_FILE_PATH)):
        os.makedirs(os.path.dirname(PDF_FILE_PATH))
        logger.info(f"Created directory: {os.path.dirname(PDF_FILE_PATH)}")

    # This part for one-time embedding and storage
    chroma_db_initialized = os.path.exists(os.path.join(CHROMA_DB_PATH, "chroma.sqlite3"))

    if not chroma_db_initialized:
        print("\n--- Initializing ChromaDB and Embedding PDF ---")
        logger.info("ChromaDB not found or not fully initialized. Performing initial embedding and storage.")
        if not os.path.exists(PDF_FILE_PATH):
            logger.error(f"PDF file '{PDF_FILE_PATH}' does not exist. Please provide a valid PDF file and run again.")
            exit(1)

        pdf_pages_text = extract_text_from_pdf(PDF_FILE_PATH)
        if not pdf_pages_text:
            logger.error("No text extracted from PDF. Exiting initial setup.")
            exit(1)
        
        semantic_chunks = create_semantic_chunks(pdf_pages_text)
        # Pass the embedding_model_name to the function
        chroma_collection = embed_and_store_chunks(semantic_chunks, CHROMA_DB_PATH, COLLECTION_NAME, embedding_model)
    else:
        logger.info("ChromaDB found. Skipping initial embedding and loading existing collection.")
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
        # Instantiate the custom embedding function for the existing collection
        gemini_ef = GeminiEmbeddingFunction(model_name=embedding_model)
        try:
            collection_list = client.list_collections()
            if any(c.name == COLLECTION_NAME for c in collection_list):
                 chroma_collection = client.get_collection(
                    name=COLLECTION_NAME,
                    embedding_function=gemini_ef
                )
                 logger.info(f"Loaded existing ChromaDB collection '{COLLECTION_NAME}' with {chroma_collection.count()} items.")
            else:
                logger.warning(f"ChromaDB directory exists, but collection '{COLLECTION_NAME}' not found. Re-initializing collection.")
                pdf_pages_text = extract_text_from_pdf(PDF_FILE_PATH)
                if not pdf_pages_text:
                    logger.error("No text extracted from PDF. Exiting initial setup.")
                    exit(1)
                semantic_chunks = create_semantic_chunks(pdf_pages_text)
                chroma_collection = embed_and_store_chunks(semantic_chunks, CHROMA_DB_PATH, COLLECTION_NAME, embedding_model)

        except Exception as e:
            logger.exception(f"Error loading existing ChromaDB collection '{COLLECTION_NAME}': {e}")
            chroma_collection = None


    # This part for continuous querying and answering
    if chroma_collection:
        print("\n--- ASK_PDF_AI Assistant ---")
        print("Type your questions about the resume. Type 'exit' to stop.")
        while True:
            query = input("\nYour question: ")
            if query.lower() == 'exit':
                print("Exiting AI Assistant. Goodbye!")
                break
            
            # 1. Retrieve relevant chunks
            retrieved_results = query_chroma_db(chroma_collection, query, embedding_model_name=embedding_model)
            
            if retrieved_results and retrieved_results['documents'] and len(retrieved_results['documents'][0]) > 0:
                # Extract only the content of the top retrieved chunks
                retrieved_chunks_content = retrieved_results['documents'][0]
                
                # 2. Generate answer using Gemini, grounded by the retrieved chunks
                gemini_answer = generate_gemini_response(query, retrieved_chunks_content)
                
                print("\nAssistant's Answer:")
                print(gemini_answer)
                
                # Optional: Print supporting chunks to log file for debugging/analysis, not to console for user
                logger.debug("\n--- Supporting Chunks (for debug/log) ---")
                for i in range(len(retrieved_results['documents'][0])):
                    doc = retrieved_results['documents'][0][i]
                    meta = retrieved_results['metadatas'][0][i]
                    dist = retrieved_results['distances'][0][i]
                    logger.debug(f"\n  Chunk {i+1} (Distance: {dist:.4f}) [Page: {meta.get('page', 'N/A')}, Paragraph: {meta.get('paragraph', 'N/A')}]")
                    logger.debug(f"  Content: {doc[:500]}...") # Log more of the content for debugging
            else:
                print("Assistant's Answer: I couldn't find relevant information in the document to answer your query.")
                logger.info(f"No relevant chunks found for query: '{query}'")
    else:
        print("\nError: Cannot proceed with querying as ChromaDB collection is not available. Check logs for details.")
        logger.critical("ChromaDB collection was not initialized/loaded successfully. Cannot start Q&A loop.")