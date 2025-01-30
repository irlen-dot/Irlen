from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from typing import List, Dict
import os

class ParagraphGenerator:
    def __init__(self, api_key):
        """
        Initialize the generator with OpenAI API key
        """
        os.environ["OPENAI_API_KEY"] = api_key
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        
        # Create a prompt template with global context
        template = """
        Global Topic: {global_topic}
        
        Task: Generate an academic paragraph addressing the following specific question within the context 
        of the global topic, using the provided resources.
        
        Specific Question: {question}
        
        Resources:
        {resources}
        
        Context and Guidelines:
        - Consider how this specific aspect relates to the global topic
        - Write in formal academic style
        - Include proper citations
        - Maintain objective tone
        - Support arguments with evidence from resources
        - Use appropriate academic vocabulary
        - Keep to one cohesive paragraph
        - Make clear connections between the specific question and the global topic
        
        Academic Paragraph:
        """
        
        self.prompt = ChatPromptTemplate.from_template(template)
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def generate_paragraph(self, global_topic: str, question: str, resources: str) -> str:
        """
        Generate an academic paragraph based on the global topic, specific question, and resources
        
        Args:
            global_topic (str): The overarching research topic
            question (str): The specific research question or subtopic
            resources (str): Relevant resources and references
            
        Returns:
            str: Generated academic paragraph
        """
        
        try:
            result = self.chain.invoke({
                "global_topic": global_topic,
                "question": question,
                "resources": resources
            })
            return result['text'].strip()
        except Exception as e:
            return f"Error generating paragraph: {str(e)}"

    def generate_multiple_paragraphs(self, topic_structure: Dict) -> Dict[str, str]:
        """
        Generate multiple paragraphs based on a structured topic dictionary
        
        Args:
            topic_structure: Dictionary containing global topic, subtopics, and resources
            Format:
            {
                "global_topic": "Main research topic",
                "subtopics": [
                    {
                        "question": "Specific question 1",
                        "resources": "Related resources"
                    },
                    ...
                ]
            }
            
        Returns:
            Dict[str, str]: Dictionary of generated paragraphs keyed by questions
        """
        results = {}
        global_topic = topic_structure["global_topic"]
        
        for subtopic in topic_structure["subtopics"]:
            paragraph = self.generate_paragraph(
                global_topic=global_topic,
                question=subtopic["question"],
                resources=subtopic["resources"]
            )
            results[subtopic["question"]] = paragraph
            
        return results
