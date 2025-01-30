from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import Dict, List

from langchain_openai import ChatOpenAI

class QuestionGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4")
        
        self.main_question_prompt = PromptTemplate(
            input_variables=["topic"],
            template="""
            Given the topic area: {topic}
            
            Generate a compelling main research question that:
            1. Is specific and focused
            2. Has depth for analysis
            3. Can be broken down into sub-questions
            4. Is debatable or requires analysis
            
            The question should be in the format:
            "How/Why/What [specific aspect] impacts/influences/affects [specific area]?"
            """
        )
        
        self.sub_questions_prompt = PromptTemplate(
            input_variables=["main_question"],
            template="""
            For the main research question: {main_question}
            
            Generate 3-5 sub-questions that:
            1. Help build understanding of the main question
            2. Follow a logical progression:
               - Definition/Context
               - Problem/Impact Analysis
               - Solutions/Implications
            3. Can each be answered in 1-2 paragraphs
            
            Return ONLY the sub-questions, one per line, with no numbering or additional text.
            """
        )
        
    def generate_question_hierarchy(self, main_question: str) -> Dict[str, List[str]]:
        """
        Generates a main question and related sub-questions
        
        Args:
            topic: The general topic area
            
        Returns:
            Dictionary containing main question and list of sub-questions
        """
        
        # Generate sub-questions
        sub_questions_chain = LLMChain(llm=self.llm, prompt=self.sub_questions_prompt)
        sub_questions = sub_questions_chain.run(main_question=main_question).splitlines()
        
        return sub_questions

        
        
    def validate_questions(self, questions: Dict[str, List[str]], resources: List[str]) -> bool:
        """
        Validates that questions can be answered with available resources
        """
        # Implementation would check resource coverage
        pass