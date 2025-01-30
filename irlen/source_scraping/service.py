from langchain.retrievers import MultiQueryRetriever
from langchain.prompts import PromptTemplate
from typing import List, Dict
import re
from langchain_community.vectorstores import Chroma

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from irlen.embed.service import Embed

class SourceScraping:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4-0125-preview")
        self.vector_store = Chroma(
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
            collection_name=Embed().get_collection_name(),
            persist_directory=Embed().get_persist_directory()
        )
        
        # Prompt for query decomposition
        self.decomposition_prompt = PromptTemplate(
            input_variables=["question"],
            template="""
            Break down this question into specific search queries that will help find relevant information:
            Question: {question}
            
            Consider:
            1. Core concepts mentioned
            2. Related terminology
            3. Specific aspects to look for
            
            Format each query to be specific and focused.
            """
        )
        
        # Prompt for relevance validation
        self.validation_prompt = PromptTemplate(
            input_variables=["question", "chunk"],
            template="""
            Question: {question}
            Text Chunk: {chunk}
            
            Determine if this chunk is directly relevant to the question by checking:
            1. Does it contain information that directly addresses the question?
            2. Is the information in the correct context?
            3. Does it provide substantive information rather than just mentioning keywords?
            
            Return ONLY 'relevant' or 'not_relevant'
            """
        )

    def retrieve_precise_chunks(self, question: str, top_k_chunks: int = 5) -> List[Dict]:
        """
        Retrieves and validates relevant chunks for a specific question
        
        Args:
            question: The specific question to find information for
            top_k_chunks: Number of chunks to retrieve per query
            
        Returns:
            List of validated chunks with metadata
        """
        print("Starting to retrieve precise chunks...")
        # 1. Generate multiple search queries
        retriever = MultiQueryRetriever.from_llm(
            retriever=self.vector_store.as_retriever(search_kwargs={"k": top_k_chunks}),
            llm=self.llm
        )
        
        # 2. Get initial chunks
        initial_chunks = retriever.get_relevant_documents(question)
        
        # 3. Validate chunks
        validated_chunks = []
        for chunk in initial_chunks:
            is_relevant = self._validate_chunk_relevance(question, chunk.page_content)
            if is_relevant:
                validated_chunks.append({
                    'content': chunk.page_content,
                    'metadata': chunk.metadata,
                    'relevance_score': self._calculate_relevance_score(question, chunk)
                })

        print("Chunks validated.")
        
        # 4. Sort by relevance
        validated_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)

        print("Chunks sorted by relevance.")

        return validated_chunks

    def _validate_chunk_relevance(self, question: str, chunk: str) -> bool:
        """Validates if a chunk is truly relevant to the question"""
        validation_response = self.llm.invoke(self.validation_prompt.format(
            question=question,
            chunk=chunk
        ))
        return validation_response.content.strip().lower() == 'relevant'

    def _calculate_relevance_score(self, question: str, chunk: str) -> float:
        """
        Calculates a relevance score based on semantic similarity between question and chunk
        
        Args:
            question: The user's question
            chunk: The document chunk to score
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Get embeddings for both texts
        question_embedding = self.vector_store._embedding_function.embed_query(question)
        chunk_embedding = self.vector_store._embedding_function.embed_query(chunk.page_content)
        
        # Calculate cosine similarity
        similarity = self._cosine_similarity(question_embedding, chunk_embedding)
        return similarity

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        return dot_product / (norm1 * norm2)

    def format_validated_chunks(self, chunks: List[Dict]) -> str:
        """Formats validated chunks with citations for use"""
        formatted_text = ""
        for chunk in chunks:
            formatted_text += f"\nSource: {chunk['metadata']['source']}\n"
            formatted_text += f"Content: {chunk['content']}\n"
            formatted_text += f"Page: {chunk['metadata'].get('page', 'N/A')}\n"
            formatted_text += "-" * 50 + "\n"
        return formatted_text