import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import torch

class VectorBuilder:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="dmis-lab/biobert-base-cased-v1.2",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
    
    def prepare_documents(self):
        documents = []
        for _, row in self.df.iterrows():
            content = f"Question: {row['Questions']}\nAnswer: {row['Answers']}"
            
            # Update metadata to match your CSV columns
            doc = Document(
                page_content=content,
                metadata={
                    'focus': row['Focus'],
                    'question': row['Questions'],
                    'source': 'MedQuAD'
                }
            )
            documents.append(doc)
        return documents
     
    def create_chunks(self, documents):
        """Split documents into chunks."""
        return self.text_splitter.split_documents(documents)
    
    def create_vectorstore(self, chunks, persist_directory="medical_vectorstore"):
        """Create and persist vector store."""
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        vectorstore.persist()
        return vectorstore

def main():
    # Initialize processor
    processor = VectorBuilder("./processed_medquad.csv")
    # Create documents with metadata
    print("Preparing documents...")
    documents = processor.prepare_documents()
    
    # Split into chunks
    print("Creating chunks...")
    chunks = processor.create_chunks(documents)
    
    # Create vector store
    print("Creating vector store...")
    vectorstore = processor.create_vectorstore(chunks)
    
    print(f"Processing complete. Total chunks created: {len(chunks)}")
    return vectorstore

if __name__ == "__main__":
    main()