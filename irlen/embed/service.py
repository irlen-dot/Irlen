from typing import List
import uuid
from langchain_openai import OpenAIEmbeddings
import chromadb
import os

class Embed:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        workdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        chroma_path = os.path.join(workdir, "chroma_db")
        self.client = chromadb.PersistentClient(path=chroma_path)
        # Create a new collection name that includes the model name to avoid dimension conflicts
        self.collection_name = "embeddings_text_embedding_3_small"

    def get_persist_directory(self):
        workdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        chroma_path = os.path.join(workdir, "chroma_db")
        return chroma_path

    def get_collection_name(self):
        """Get the collection name.
        
        Returns:
            Collection name
        """
        return self.collection_name
        
    def get_chroma_client(self):
        """Get the ChromaDB client instance.
        
        Returns:
            ChromaDB client instance
        """
        return self.client

    def embed_chunks(self, documents: List[dict[str, str]]) -> None:
        """Store embeddings for input documents using OpenAI embeddings.
        
        Args:
            documents: List of dictionaries containing text and id
        """

        print("Embedding chunks...")

        for doc in documents:
            embedding = self.embeddings.embed_query(doc["text"])

            embedded_doc = {
                "id": str(uuid.uuid4()),
                "text": doc["text"], 
                "embedding": embedding,
                "metadata": {
                    "title": doc["file_title"],
                }
            }
            self.store_embeddings(embedded_doc)
            print("Embedded chunk.")


        print("Done embedding chunks")



    def embed(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)
    
    
    def store_embeddings(self, document: dict[str, List[float]]):
        """Store embeddings in the ChromaDB database.
        
        Args:
            documents: List of dictionaries containing embeddings and text
        """
        collection = self.client.get_or_create_collection(self.collection_name)
        
        print(document["metadata"]["title"])
        
        collection.add(
            embeddings=document["embedding"],
            documents=document["text"],
            ids=document["id"],
            metadatas={"title": document["metadata"]["title"]},
        )


    

    def get_embeddings(self, query: str) -> List[float]:
        """Get embeddings for a query using the ChromaDB database.
        
        Args:
            query: Input query to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        collection = self.client.get_collection(self.collection_name)
        embedding = self.embed(query)   
        results = collection.query(query_embeddings=[
            embedding
        ], n_results=1)
        return results["documents"]