import os
from typing import List
import uuid
# from langchain_core import SentenceSplitter
from langchain_text_splitters import TokenTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


load_dotenv()

class Chunking:
    def __init__(self):
        pass

    def _get_files(self) -> List[str]:
        """Get list of files from resources folder"""
        import os
        
        resources_dir = os.path.join(os.getcwd(), 'resources')
        files = []
        
        for file in os.listdir(resources_dir):
            if os.path.isfile(os.path.join(resources_dir, file)):
                files.append(file)
                
        return files


    def get_chunks(self, chunk_size: int = 256, overlap_pct: float = 0.1) -> List[dict]:
        """Get chunks and their embeddings from input text.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            overlap_pct: Percentage of overlap between chunks (0.0-1.0)
            
        Returns:
            List of dicts containing chunk text, file title, and embedding
        """
        # Get list of files
        files = self._get_files()
        
        # Initialize results list
        chunks_with_metadata = []
        
        # Process each file separately
        resources_dir = os.path.join(os.getcwd(), 'resources')

        for file in files:
            file_path = os.path.join(resources_dir, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            # Get text chunks for this file
            file_chunks = self._chunk(text, chunk_size, overlap_pct)
            
            # Add file title to each chunk
            for chunk in file_chunks:
                chunks_with_metadata.append({
                    'text': chunk,
                    'file_title': file
                })

        return chunks_with_metadata

        

    def _chunk(self, text: str, chunk_size: int = 256, overlap_pct: float = 0.1) -> List[str]:
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
            # separator="."
        )


        # Split text into chunks
        chunks = splitter.split_text(text)
        return chunks