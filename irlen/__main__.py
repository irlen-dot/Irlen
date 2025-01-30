from irlen.chunking.service import Chunking
from irlen.embed.service import Embed
from irlen.question_generator.service import QuestionGenerator
from irlen.source_scraping.service import SourceScraping

def process_text():
    """Process text by chunking it, embedding chunks, and storing in database.
    
    Args:
        text: Input text to process
        chunk_size: Maximum size of each chunk in characters (default 256)
        overlap_pct: Percentage of overlap between chunks (default 0.1)
        
    Returns:
        List of text chunks with embeddings
    """
    
    source_scraping = SourceScraping()
    result = source_scraping.retrieve_precise_chunks("What is Mcdonization?")
    

    print(result)

def main():
    # Example text to chunk
    sample_text = """This is a sample text that will be split into chunks.
    The chunking service will process this text and split it into smaller pieces
    based on the specified chunk size and overlap percentage."""
    
    # Set chunk parameters
    chunk_size = 100  # Smaller size for demonstration
    overlap_pct = 0.1
    
    # Process the text
    chunks = process_text(sample_text, chunk_size, overlap_pct)
    
    # Print results
    # print(f"Number of chunks: {len(chunks)}"

if __name__ == "__main__":
    main()


