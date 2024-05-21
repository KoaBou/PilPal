import os
import dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.chains.conversation.base import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from typing import Dict, List, Tuple

dotenv.load_dotenv()


class Chatbot():
    def __init__(self, patient_dct):
        self.llm = ChatOpenAI(
            temperature=1,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        self.memory = ConversationSummaryMemory(
            llm=self.llm,
            memory_key="history"
        )

        self.patient_info = self.pi2text(patient_dct)

        self.create_chain()

    def pi2text(self, patient_dct: Dict) -> str:
        patient_info = ""

        for key, value in patient_dct.items():
            patient_info += f'{key}: {value}\n'

        return patient_info

    def create_chain(self):
        template = """You are a healthcare AI responsible for addressing patients' inquiries. Provide personalized advice based on their personal information. Ensure your responses cite sources.
Patient Information:
""" + self.patient_info + """
This is a conversation between you and the patient:
{history}
Human: {input}
AI:"""

        prompt = PromptTemplate(
            input_variables=['history', 'input'],
            template=template,
        )

        self.chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=True,
            memory=self.memory,
        )


if __name__ == '__main__':

    patient_info = {
        'name' : 'Nguyen Trung Nguyen',
        'gender' : 'Male',
        'medical record' : 'Blood Clots'
    }

    query = "Can I use Paracetamol?"

    chatbot = Chatbot(patient_dct=patient_info)

    print(chatbot.chain.invoke(
        {
            'input': query,
        }
    ))

