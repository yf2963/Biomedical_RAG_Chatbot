from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import torch
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from SimilarQuestionRetriever import SimilarQuestionRetriever

class MedicalRAGSystem:
    def __init__(self, model_path="BioMistral/BioMistral-7B", persist_directory="medical_vectorstore"):
        self.setup_model(model_path)
        self.load_vectorstore(persist_directory)
        self.setup_prompt()
        self.setup_qa_chain()

    def setup_model(self, model_path):
        """Initialize BioMistral model and tokenizer."""
        print("Loading BioMistral model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,  # Back to max_length for better control
            temperature=0.1,  # Reduced for more focused responses
            do_sample=False,  # Deterministic output
            top_p=0.95,
            repetition_penalty=1.1  # Prevent repetition
        )
        
        self.llm = HuggingFacePipeline(pipeline=pipeline)
        print("Model setup complete!")

    def load_vectorstore(self, persist_directory):
        """Load existing vector store with similar question retrieval."""
        print("Loading vector store...")
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=HuggingFaceEmbeddings(
                model_name="dmis-lab/biobert-base-cased-v1.2",
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
            )
        )
        self.retriever = SimilarQuestionRetriever(vectorstore=self.vectorstore)
        print("Vector store loaded!")

    def setup_prompt(self):
        """Create enhanced prompt template for medical QA."""
        template = """[INST] You are a medical assistant answering health-related questions.
        Use only the following medical information to answer the question.
        If no relevant information is found in the context, respond with:
        "I apologize, but I don't have any information about [topic] in my knowledge base."
        
        Medical Knowledge:
        {context}
        
        Question: {question}
        
        Please provide a clear, accurate medical response using only the information from the Medical Knowledge section above. [/INST]"""
        
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def setup_qa_chain(self):
        """Setup the question-answering chain."""
        print("Setting up QA chain...")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={
                "prompt": self.prompt
            },
            return_source_documents=True
        )
        print("QA chain ready!")

    def get_answer(self, question: str):
        """Get answer for a medical question."""
        try:
            result = self.qa_chain({"query": question})
            return {
                "answer": result["result"],
                "sources": [
                    {
                        "focus": doc.metadata.get("focus", ""),
                        "question": doc.metadata.get("question", "")
                    } 
                    for doc in result["source_documents"]
                ]
            }
        except Exception as e:
            return {
                "answer": "I encountered an error processing your question. Please try again.",
                "error": str(e)
            }
        
rag_system = MedicalRAGSystem()
def test_medical_qa(question: str, rag_system):
    print("Question:", question)
    print("\nProcessing...")
    response = rag_system.get_answer(question)
    
    if "error" in response:
        print("\nError:", response["error"])
        return
        
    answer = response.get("answer", "")
    if "[INST]" in answer:
        answer = answer.split("[/INST]")[-1].strip()
    print("\nAnswer:", answer)
    
    if "sources" in response:
        print("\nSources:")
        for source in response["sources"]:
            print(f"\nFocus: {source['focus']}")
            print(f"Original Question: {source['question']}")

test_medical_qa("How to diagnose Schimke immunoosseous dysplasia ?",rag_system)