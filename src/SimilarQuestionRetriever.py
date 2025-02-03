from xml.dom.minidom import Document
from pydantic import Field
from langchain.schema import BaseRetriever
from typing import List, Any

class SimilarQuestionRetriever(BaseRetriever):
    vectorstore: Any = Field(default=None, description="Vector store for document retrieval")
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Get the most similar document
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=1)
        
        if not docs_and_scores:
            return []
        
        most_similar_doc, _ = docs_and_scores[0]
        original_question = most_similar_doc.metadata.get('question', '')
        
        # Get documents with the same question and their answers
        similar_docs = self.vectorstore.similarity_search(
            original_question,
            k=2,
            filter={"question": original_question}
        )
        
        return similar_docs

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)