import os
import shutil
import re
import json

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.schema.runnable import RunnableMap


from utils import get_user_repos


class Links(BaseModel):
    """
    Pydantic model representing a user's GitHub link.

    Attributes:
        github_link (str | None): A valid URL linking to the user's GitHub profile.
    """
    github_link: str | None = Field(description="""This property must contain a valid URL linking to the user's GitHub profile. The URL should be formatted correctly to ensure it points directly to the user's GitHub profile page.
Example: 'https://github.com/username'""")


class Questions(BaseModel):
    """
    Pydantic model representing questions derived from a job description.

    Attributes:
        questions (list[str]): A list of queries based on the job description.
    """
    questions: list[str] = Field(description="""This property should contain a list of queries based on the job description. Example: 'Does the candidate have 3+ years of experience?'""")


class Answer(BaseModel):
    """
    Pydantic model representing answers to questions about a candidate's resume.

    Attributes:
        answer (str): The answer to the question asked.
        score (int): A number (0 or 1) based on the answer.
        summary (str): A summarized version of the context provided in the prompt.
    """
    answer: str = Field(description="""This property should contain the answer of the question asked.Should be either 'Yes' or 'No'""")
    score: int = Field(description="""This property should contain a number between 0 and 1 based on the suitability of the context for the question.""")
    binary_score: int = Field(description="""This property should contain a number 0 or 1 corresponding to the answer 'No' or 'Yes'.""")
    summary: str = Field(description="""This property should contain a summarized version of the context provided in the prompt.""")


class ResumeAgent:
    """
    A class to process and analyze resumes in PDF format.

    Attributes:
        cv_dir (str): Directory containing PDF resumes.
        pdf_cv (list): List of PDF files in the directory.
        embedding_model: Model for generating text embeddings.
        link_chain: Chain for extracting GitHub links from resumes.
        query_chain: Chain for generating questions from job descriptions.
        qa_chain: Chain for answering questions based on resume context.
        contrastive_model: Model for computing contrastive similarity scores.

    Methods:
        __init_embedding_model(): Initializes the embedding model.
        __init_link_chain(): Initializes the link extraction chain.
        __init_query_chain(): Initializes the query generation chain.
        __init_qa_chain(): Initializes the question-answering chain.
        __load_and_split(pdf_path): Loads and splits PDF documents.
        __init_chroma(collection_name, docs, persist_directory='./chroma'): Initializes the Chroma vector store.
        __fn_to_collection(file_name): Converts filename to collection name.
        get_links(): Extracts GitHub links from resumes.
        __scrape_links(link, github_link=False): Scrapes GitHub links.
        __embed_text(collection_name, text, metadata): Embeds text into the Chroma vector store.
        analyze(job_description: str): Analyzes resumes against the job description.
    """
    def __init__(self, cv_dir):
        """
        Initializes the ResumeAgent with the directory containing PDF resumes.

        Args:
            cv_dir (str): Directory containing PDF resumes.
        """
        self.cv_dir = cv_dir
        
        # Get all PDF files in the directory
        files = os.listdir(cv_dir)
        self.pdf_cv = [file for file in files if file.lower().endswith('.pdf')]
        
        # Initialize embedding model
        self.__init_embedding_model()

        # Process each PDF file
        for file_name in self.pdf_cv:
            path = os.path.join(cv_dir, file_name)
            docs = self.__load_and_split(path)
            collection_name = self.__fn_to_collection(file_name)
            self.__init_chroma(collection_name, docs)

        self.__init_link_chain()
        
    def __init_embedding_model(self):
        """
        Initializes the embedding model using HuggingFace embeddings.
        """
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def __init_link_chain(self):
        """
        Initializes the link extraction chain using ChatGroq and JsonOutputParser.
        """
        self.llm = ChatGroq(
            temperature=0,
            model='llama3-70b-8192'
        )

        output_parser = JsonOutputParser(pydantic_object=Links)

        template = """You are an assistant for question-answering tasks based on the context of the given resume. Use only the following pieces of retrieved context to answer the question.
        Answer questions based on the format instruction given below. Do not provide any introductory text, explanatory text or notes in the result.
        \nQuestion: {question} \nContext: {context} \n format instruction: {format_instructions} \nAnswer:"""

        prompt = ChatPromptTemplate.from_template(template)

        self.link_chain = RunnableMap({
            "context": lambda x: x["retriever"].invoke(x["question"]),
            "question": lambda x: x["question"],
            "format_instructions": lambda x: output_parser.get_format_instructions()
        }) | prompt | self.llm | output_parser

    def __init_query_chain(self):
        """
        Initializes the query generation chain using ChatGroq and JsonOutputParser.
        """
        output_parser = JsonOutputParser(pydantic_object=Questions)

        template = """List the requirements mentioned in the job description as a descriptive question. \n job description: {job_description}  \n format instruction: {format_instructions}:"""

        prompt = ChatPromptTemplate.from_template(template)

        self.query_chain = RunnableMap({
            "job_description": lambda x: x["job description"],
            "format_instructions": lambda x: output_parser.get_format_instructions()
        }) | prompt | self.llm | output_parser

    def __init_qa_chain(self):
        """
        Initializes the question-answering chain using ChatGroq and JsonOutputParser.
        """
        output_parser = JsonOutputParser(pydantic_object=Answer)

        template = """You are an expert at evaluating a candidate based on the context of the given resume. Use only the following pieces of retrieved context to generate the answer to the question.
        Format your answer based on the format instruction given below. Do not provide any introductory text, explanatory text or notes in the result. The question should be carefully considered to determine appropriate answer from the context.
        \nQuestion: {question} \nContext: {context} \n format instruction: {format_instructions}"""

        prompt = PromptTemplate.from_template(template)

        self.qa_chain = RunnableMap({
            "question": lambda x: x["question"],
            "context": lambda x: x["retriever"].invoke(x["question"]),
            "format_instructions": lambda x: output_parser.get_format_instructions()
        }) | prompt | self.llm | output_parser

    def __load_and_split(self, pdf_path):
        """
        Loads and splits PDF documents into chunks.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            list: List of document chunks.
        """
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100, add_start_index=True
        )
        docs = text_splitter.split_documents(pages)

        return docs

    def __init_chroma(self, collection_name, docs, persist_directory='./chroma'):
        """
        Initializes the Chroma vector store with the provided documents.

        Args:
            collection_name (str): Name of the collection.
            docs (list): List of document chunks.
            persist_directory (str, optional): Directory to persist the vector store. Defaults to './chroma'.
        """
        vectorstore = Chroma.from_documents(
            docs,
            self.embedding_model,
            collection_name=collection_name,
            persist_directory=persist_directory
        )

        del vectorstore

    def __fn_to_collection(self, file_name):
        """
        Converts a filename to a collection name.

        Args:
            file_name (str): Name of the file.

        Returns:
            str: Collection name derived from the filename.
        """
        return file_name.split('.')[0].replace(" ", "")

    def get_links(self):
        """
        Extracts GitHub links from resumes.

        Returns:
            list: List of GitHub links.
        """
        prompt = [
            {
                "question": "What is the candidate's github profile?",
                "retriever": Chroma(
                    collection_name=self.__fn_to_collection(file_name),
                    persist_directory='./chroma',
                    embedding_function=self.embedding_model
                ).as_retriever()
            }
            for file_name in self.pdf_cv
        ]

        return self.link_chain.batch(prompt)
    
    def __scrape_links(self, link, github_link=False):
        """
        Scrapes GitHub links to extract repository information.

        Args:
            link (str): GitHub link.
            github_link (bool, optional): Flag indicating if the link is a GitHub link. Defaults to False.

        Returns:
            tuple: Repository text and metadata.
        """
        if github_link:
            pattern = r"https?://github\.com/([^/]+)"
            match = re.match(pattern, link)
            
            if match:
                username = match.group(1)
                repo_info = get_user_repos(username)
                
                if 'error' in repo_info:
                    return None, None
                
                repo_text = '\n'.join([str(info) for info in repo_info])
                metadata = {
                    'source': github_link,
                    'description': 'Github Projects in JSON format'
                }
                return repo_text, metadata
            else:
                return None, None
        else:
            return None, None
    
    def __embed_text(self, collection_name, text, metadata):
        """
        Embeds text into the Chroma vector store.

        Args:
            collection_name (str): Name of the collection.
            text (str): Text to be embedded.
            metadata (dict): Metadata for the text.
        """
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        documents = text_splitter.create_documents(text, metadata)
        vector_store = Chroma(
            collection_name=collection_name,
            persist_directory='./chroma',
            embedding_function=self.embedding_model
        )
        vector_store.add_documents(documents)

    def analyze(self, job_description: str):
        """
        Analyzes resumes against the job description.

        Args:
            job_description (str): Job description text.

        Yields:
            tuple: Filename and analysis results for each resume.
        """
        doc_links = self.get_links()

        for doc_link, file_name in zip(doc_links, self.pdf_cv):
            if doc_link['github_link'] is not None:
                text, metadata = self.__scrape_links(doc_link['github_link'], github_link=True)
                print(f"file: {file_name}, text: {text}")
                if text is not None:
                    try:
                        self.__embed_text(self.__fn_to_collection(file_name), text, metadata)
                    except Exception as e:
                        print(e)

        self.__init_query_chain()
        result = self.query_chain.invoke({
            'job description': job_description
        })

        self.__init_qa_chain()

        for file_name in self.pdf_cv:
            retriever = Chroma(
                collection_name=self.__fn_to_collection(file_name),
                persist_directory='./chroma',
                embedding_function=self.embedding_model
            ).as_retriever()
            
            print(f'Made retriever for {file_name}')
            
            for question in result['questions']:
                answer = self.qa_chain.invoke(
                    {
                        "retriever": retriever,
                        "question": question
                    }
                )

                yield file_name, {
                    'question': question,
                    'answer': answer['answer'],
                    'remark': answer['summary'],
                    'LLM score': answer['score'],
                    'Binary score': answer['binary_score'],
                }

    def __del__(self):
        """
        Clean up Chroma directory on deletion of the ResumeAgent instance.
        """
        if os.path.exists('./chroma'):
            shutil.rmtree('./chroma')
