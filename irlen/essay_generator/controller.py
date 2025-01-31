from typing import Dict
from irlen.essay_generator.body_paragraph import BodyParagraphGenerator
from irlen.essay_generator.intro import IntroductionGenerator
from irlen.essay_generator.summary import SummaryGenerator
from irlen.question_generator.service import QuestionGenerator
from irlen.source_scraping.service import SourceScraping


class EssayGeneratorController:
    def __init__(self):
        self.body_paragraph_generator = BodyParagraphGenerator()
        self.question_generator = QuestionGenerator()
        self.introduction_generator = IntroductionGenerator()
        self.source_scraping = SourceScraping()
        self.summary_generator = SummaryGenerator()
        pass




    def generate_essay(self, global_topic: str) -> Dict[str, str]:
        questions = self.question_generator.generate_question_hierarchy(global_topic)
        
        # print(questions)
        introduction = self._generate_introduction(global_topic, questions)
        
        body = self._generate_body(global_topic, questions)

        theses = [paragraph["thesis"] for paragraph in body]
        summary = self.summary_generator.generate_summary(global_topic, theses)

        essay = self.build_essay(introduction, body, summary)

        return essay

    def build_essay(self, introduction: Dict[str, str], body: Dict[str, str], summary: Dict[str, str]) -> Dict[str, str]:

        essay = f"{introduction['introduction']}\n\n"
        
        for paragraph in body:
            essay += f"{paragraph['paragraph']}\n\n"
            
        essay += f"{summary['summary']}"
        return essay


    def _generate_introduction(self, global_topic: str, questions: list[str]) -> Dict[str, str]:

        introduction = self.introduction_generator.generate_introduction(global_topic, questions)
        return introduction
    
    def _generate_body(self, global_topic: str, questions: list[str]) -> Dict[str, str]:
        body_paragraphs = []
        for question in questions:
            resources = self.source_scraping.retrieve_precise_chunks(question)
            body = self.body_paragraph_generator.generate_paragraph(global_topic, question, resources)
            
            body_paragraphs.append(body)

        return body_paragraphs



