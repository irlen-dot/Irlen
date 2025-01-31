from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from typing import List, Dict
import os

from dotenv import load_dotenv

load_dotenv()

class SummaryGenerator:
    def __init__(self):

        """
        Initialize the summary generator
        

        Args:
            api_key (str): OpenAI API key
        """
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
        

        # Template for generating summary from thesis statements
        self.summary_template = """
        Task: Generate a cohesive summary paragraph that synthesizes the following thesis statements
        in the context of the global topic.

        Global Topic: {global_topic}

        Thesis Statements:
        {thesis_statements}

        Guidelines:
        - Create a flowing narrative that connects the thesis statements
        - Maintain clear relationship to the global topic
        - Keep academic tone
        - Be concise but comprehensive
        - Create smooth transitions between ideas
        - End with a statement that ties everything together
        
        Summary Paragraph:
        """
        
        self.prompt = ChatPromptTemplate.from_template(self.summary_template)
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def _format_thesis_statements(self, statements: List[str]) -> str:
        """
        Format thesis statements for the summary prompt
        
        Args:
            statements (List[str]): List of thesis statements
            
        Returns:
            str: Formatted thesis statements
        """
        return "\n".join(f"- {statement}" for statement in statements)

    def generate_summary(self, global_topic: str, thesis_statements: List[str]) -> Dict[str, str]:
        """
        Generate a summary from provided thesis statements
        
        Args:
            global_topic (str): The overarching research topic
            thesis_statements (List[str]): List of thesis statements to synthesize
            
        Returns:
            Dict[str, str]: Dictionary containing the summary and metadata
        """
        try:
            formatted_statements = self._format_thesis_statements(thesis_statements)
            
            # Generate summary
            result = self.chain.invoke({
                "global_topic": global_topic,
                "thesis_statements": formatted_statements
            })
            
            return {
                "summary": result['text'].strip(),
                "thesis_statements": thesis_statements,
                "global_topic": global_topic,
                "num_statements": len(thesis_statements),
                "status": "success"
            }
        except Exception as e:
            return {
                "summary": "",
                "thesis_statements": thesis_statements,
                "global_topic": global_topic,
                "num_statements": len(thesis_statements),
                "status": "error",
                "error_message": str(e)
            }
