from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from typing import List, Dict
import os

from dotenv import load_dotenv

load_dotenv()

class IntroductionGenerator:
    def __init__(self):
        """
        Initialize the introduction generator
        
        """

        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
        
        # Create template for introduction generation
        template = """
        Task: Generate an academic introduction paragraph that introduces the global research topic
        and smoothly transitions into the specific research questions that will be addressed.

        Global Topic: {global_topic}

        Research Questions to be Addressed:
        {questions}

        Guidelines:
        - Start with a broad context for the global topic
        - Establish the significance of the research area
        - Gradually narrow down to the specific aspects being studied
        - Introduce each research question naturally within the flow
        - Maintain formal academic tone
        - Keep to 1-2 cohesive paragraphs
        - End with a clear indication of what the paper will explore

        Academic Introduction:
        """
        
        self.prompt = ChatPromptTemplate.from_template(template)
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def _format_questions(self, questions: List[str]) -> str:
        """
        Format the list of questions for the prompt
        
        Args:
            questions (List[str]): List of research questions
            
        Returns:
            str: Formatted questions string
        """
        return "\n".join(f"- {question}" for question in questions)

    def generate_introduction(self, global_topic: str, questions: List[str]) -> Dict[str, str]:
        """
        Generate an academic introduction based on the global topic and list of questions
        
        Args:
            global_topic (str): The overarching research topic
            questions (List[str]): List of specific research questions to be addressed
            
        Returns:
            Dict[str, str]: Dictionary containing the introduction and metadata
        """
        try:
            formatted_questions = self._format_questions(questions)
            
            result = self.chain.invoke({
                "global_topic": global_topic,
                "questions": formatted_questions
            })
            
            return {
                "introduction": result['text'].strip(),
                "global_topic": global_topic,
                "questions_addressed": questions,
                "status": "success"
            }
        except Exception as e:
            return {
                "introduction": "",
                "global_topic": global_topic,
                "questions_addressed": questions,
                "status": "error",
                "error_message": str(e)
            }
