import os
from google.cloud import aiplatform
import numpy as np
import google.generativeai as genai

# Set Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/metehan/Documents/Cursor/vectorize/metehan777-e4063b146f6d.json"

# Set your Google Cloud project ID here
PROJECT_ID = "metehan777"  # Replace with your actual project ID

# Set Gemini API key
os.environ["GEMINI_API_KEY"] = "AIzaSyBWQaMs7m-8QYKkFGBC9TVc4Ajpjn6bYpY"

class GoogleCloudEmbeddings:
    def __init__(self, project_id=None, location="us-central1", model_name="models/embedding-001"):
        """
        Initialize the Google Cloud Embeddings client.
        
        Args:
            project_id: Your Google Cloud project ID
            location: Region where your model is deployed
            model_name: The embedding model name
        """
        # Use the provided project_id or fall back to the global PROJECT_ID
        self.project_id = project_id if project_id else PROJECT_ID
        self.location = location
        self.model_name = model_name
        
        if not self.project_id:
            raise ValueError("Project ID is required. Please set PROJECT_ID in the code or provide it when initializing.")
        
        # For Vertex AI (if needed for other operations)
        aiplatform.init(project=self.project_id, location=self.location)
        
        # For Gemini API
        self.api_key = os.environ["GEMINI_API_KEY"]
        
        # Initialize Gemini
        genai.configure(api_key=self.api_key)
        
    def get_embeddings(self, texts):
        """
        Get embeddings for a list of texts using Gemini embedding model.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        # Process each text individually
        for text in texts:
            try:
                # Get embedding using Gemini API
                result = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="RETRIEVAL_DOCUMENT"  # or "RETRIEVAL_QUERY" for queries
                )
                
                # Extract the embedding vector
                embedding = np.array(result["embedding"])
                embeddings.append(embedding)
                
            except Exception as e:
                print(f"Error getting embedding for text: {e}")
                # Add a zero vector as fallback
                embeddings.append(np.zeros(768))  # Default to 768 dimensions for fallback
        
        return embeddings

class EmbeddingProcessor:
    def __init__(self, embedding_client):
        self.embedding_client = embedding_client
        
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