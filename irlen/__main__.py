from irlen.chunking.service import Chunking
from irlen.embed.service import Embed
from irlen.essay_generator.body_paragraph import BodyParagraphGenerator
from irlen.essay_generator.controller import EssayGeneratorController
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
    # chunking = Chunking()
    # chunks = chunking.get_chunks()

    # embed = Embed()
    # embed.embed_chunks(chunks)


    essay_generator = EssayGeneratorController()

    result = essay_generator.generate_essay("How do female filmmakers approach the fight against male gaze?")
    print(result)



def main():
    # Process the text
    process_text()
    
    # Print results
    # print(f"Number of chunks: {len(chunks)}"

if __name__ == "__main__":
    main()


