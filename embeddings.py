import os
from google.cloud import aiplatform
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Gemini API key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# Configure Gemini with the API key
genai.configure(api_key=GEMINI_API_KEY)

class GoogleCloudEmbeddings:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=api_key)
        self.model_name = "models/gemini-embedding-exp-03-07"

    def embed_text(self, text):
        if not text.strip():
            raise ValueError("Content for embedding must not be empty.")
        embedding_response = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_document"
        )
        return embedding_response["embedding"]

class EmbeddingProcessor:
    def __init__(self, embedding_client):
        self.embedding_client = embedding_client

    def generate_embeddings(self, content):
        embeddings_data = []
        for item in content:
            if not item['content'].strip():
                # Skip empty content
                continue
            embedding = self.embedding_client.embed_text(item['content'])
            embeddings_data.append({
                'url': item['url'],
                'title': item['title'],
                'content': item['content'],
                'embedding': embedding
            })
        return embeddings_data

    def process_pages(self, pages_data):
        """
        Process crawled pages and add embeddings.
        
        Args:
            pages_data: List of dictionaries with page information
            
        Returns:
            Updated pages_data with embeddings
        """
        # Extract texts to embed
        texts = [f"{page['title']} {page['content'][:1000]}" for page in pages_data]
        
        # Get embeddings
        embeddings = self.embedding_client.get_embeddings(texts)
        
        # Add embeddings to pages_data
        for i, page in enumerate(pages_data):
            page['embedding'] = embeddings[i]
            
        return pages_data 