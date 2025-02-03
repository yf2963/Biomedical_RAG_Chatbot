import google.generativeai as genai
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)
from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
import pandas as pd
from MedicalRAGSystem import MedicalRAGSystem

rag_system = MedicalRAGSystem()

class RAGEvaluator:
    def __init__(self, rag_system, test_data_path="./processed_medquad.csv"):
        self.rag_system = rag_system
        self.test_df = pd.read_csv(test_data_path)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.1,
            google_api_key="YOUR_GOOGLE_API_KEY"
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    def prepare_evaluation_data(self, num_samples=10):
        """Prepare dataset for RAGAS evaluation"""
        eval_data = []
        
        # Sample questions
        sampled_data = self.test_df.sample(n=num_samples, random_state=42)
        
        for _, row in sampled_data.iterrows():
            # Get RAG response
            response = self.rag_system.get_answer(row['Questions'])
            
            # Clean up answer
            answer = response["answer"]
            if "[INST]" in answer:
                answer = answer.split("[/INST]")[-1].strip()
            
            # Format contexts properly for RAGAS
            contexts = []
            for doc in response['sources']:
                context = f"{doc['question']}: {doc['focus']}"
                if len(context) > 500:
                    context = context[:500] + "..."
                contexts.append(context)
            
            # Truncate reference if too long
            reference = row['Answers']
            if len(reference) > 1000:
                reference = reference[:1000] + "..."
                
            data_point = {
                "question": row['Questions'],
                "answer": answer,
                "contexts": contexts,
                "reference": reference,
            }
            
            eval_data.append(data_point)
        
        dataset = Dataset.from_list(eval_data)
        return dataset

    def run_evaluation(self, num_samples=10):
        """Run RAGAS evaluation"""
        print("Preparing evaluation dataset...")
        eval_dataset = self.prepare_evaluation_data(num_samples)
        
        print("\nRunning RAGAS evaluation...")
        results = evaluate(
            eval_dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision
            ],
            llm=self.llm,
            embeddings=self.embeddings,
            raise_exceptions=True
        )
        
        return results

    def print_results(self, results):
        """Print evaluation results in a readable format"""
        print("\nRAGAS Evaluation Results:")
        print("-" * 50)
        print(f"Results type: {type(results)}")
        print(f"Results content: {results}")
        
        if isinstance(results, list):
            for i, result in enumerate(results):
                print(f"\nResult {i}:")
                print(f"Type: {type(result)}")
                print(f"Dir: {dir(result)}")
                try:
                    # Try to access common attributes
                    if hasattr(result, 'name'):
                        print(f"Name: {result.name}")
                    if hasattr(result, 'score'):
                        print(f"Score: {result.score:.3f}")
                    if hasattr(result, 'metadata'):
                        print(f"Metadata: {result.metadata}")
                except Exception as e:
                    print(f"Error accessing result attributes: {str(e)}")

evaluator = RAGEvaluator(rag_system)
results = evaluator.run_evaluation(num_samples=3)
evaluator.print_results(results)