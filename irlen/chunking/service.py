import os
from typing import List
import uuid
# from langchain_core import SentenceSplitter
from langchain_text_splitters import TokenTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

class Chunking:
    def __init__(self):
        pass

    def get_files(self) -> List[str]:
        """Get list of files from resources folder"""
        import os
        
        resources_dir = os.path.join(os.getcwd(), 'resources')
        files = []
        
        for file in os.listdir(resources_dir):
            if os.path.isfile(os.path.join(resources_dir, file)):
                files.append(file)
                
        return files


    def get_chunks(self, text: str, chunk_size: int = 512, overlap_pct: float = 0.1) -> List[dict]:
        """Get chunks and their embeddings from input text.
        
        Args:
            text: Input text to chunk and embed
            chunk_size: Maximum size of each chunk in characters
            overlap_pct: Percentage of overlap between chunks (0.0-1.0)
            
        Returns:
            List of dicts containing chunk text and embedding
        """
        # Get list of files
        files = self.get_files()
        
        # Initialize text variable
        text = ""
        
        # Read content from each file
        resources_dir = os.path.join(os.getcwd(), 'resources')

        for file in files:
            file_path = os.path.join(resources_dir, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                text += f.read() + "\n"

        # Get text chunks
        chunks = self.chunk(text, chunk_size, overlap_pct)

        
   
        return chunks

        

    def chunk(self, text: str, chunk_size: int = 512, overlap_pct: float = 0.1) -> List[str]:
        """Split text into overlapping chunks using sentence boundaries.
        
        Args:
            text: Input text to chunk
            chunk_size: Maximum size of each chunk in characters (default 512)
            overlap_pct: Percentage of overlap between chunks (default 0.1)
            
        Returns:
            List of text chunks
        """
        # Calculate overlap size in characters
        overlap_size = int(chunk_size * overlap_pct)
        
        # Initialize sentence splitter with correct chunk size and overlap
        splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
            # separators=[".", "?", "!",]
        )
        
        # Split text into chunks
        chunks = splitter.split_text(text)
        return chunks