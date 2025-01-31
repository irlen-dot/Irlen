from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from typing import Dict
import os

from dotenv import load_dotenv

load_dotenv()

class BodyParagraphGenerator:
    def __init__(self):
        """

        Initialize the generator with OpenAI API key
        """
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
        


        # Create a prompt template with global context
        template = """
        Global Topic: {global_topic}
        
        Task: Generate an academic paragraph addressing the following specific question within the context 
        of the global topic, using the provided resources. Then provide a concise thesis statement
        (maximum 10 words) that captures the core argument.
        
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
        
        Please format your response as follows:
        
        PARAGRAPH:
        [Your academic paragraph here]
        
        THESIS:
        [Core argument in 10 words or less]
        """
        
        self.prompt = ChatPromptTemplate.from_template(template)
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def generate_paragraph(self, global_topic: str, question: str, resources: str) -> Dict[str, str]:
        """
        Generate an academic paragraph based on the global topic, specific question, and resources
        
        Args:
            global_topic (str): The overarching research topic
            question (str): The specific research question or subtopic
            resources (str): Relevant resources and references
            
        Returns:
            Dict[str, str]: Dictionary containing the generated paragraph and thesis
        """
        
        try:
            result = self.chain.invoke({
                "global_topic": global_topic,
                "question": question,
                "resources": resources
            })
            
            # Split the response into paragraph and thesis
            text = result['text'].strip()
            parts = text.split('THESIS:')
            
            paragraph = parts[0].replace('PARAGRAPH:', '').strip()
            thesis = parts[1].strip() if len(parts) > 1 else "Thesis not found"
            
            return {
                "paragraph": paragraph,
                "thesis": thesis
            }
        except Exception as e:
            return {
                "paragraph": f"Error generating paragraph: {str(e)}",
                "thesis": "Error generating thesis"
            }


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
