# Biomedical RAG Chatbot

A biomedical question-answering system built on the MedQuAD dataset using RAG (Retrieval-Augmented Generation) with BioMistral and BioBERT embeddings. This system provides accurate medical information by leveraging a large dataset of medical Q&A pairs from trusted NIH sources.

## Overview

This project implements a medical domain question-answering system using the following components:

- **MedQuAD Dataset**: A comprehensive medical Q&A dataset containing 47,457 pairs from NIH websites
- **BioMistral-7B**: Large Language Model fine-tuned for biomedical text generation
- **BioBERT**: Domain-specific BERT model used for generating embeddings
- **ChromaDB**: Vector store for efficient similarity search
- **LangChain**: Framework for building the RAG pipeline

## Development Environment

The project can be run in two environments:

1. **Local Environment**
   - Suitable for development and testing
   - Requires sufficient RAM and storage
   
2. **Kaggle Notebooks (Recommended)**
   - Recommended for speed, making vector and inference
   - Provides free GPU access (T4 x2) for 30/hours a week
   - Better performance for compute-intensive tasks

## Requirements

```
transformers
torch
langchain
langchain-community
chromadb
pandas
numpy
xmltodict
jsonpath
ragas
sentence-transformers
google-generativeai
langchain-google-genai
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yf963/Biomedical_RAG_Chatbot.git
cd Biomedical_RAG_Chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Processing Pipeline

### 1. XML Preprocessing
The first step involves processing the raw MedQuAD XML files into a structured CSV format:

```python
from src.preprocessing import processXmlFile

# Excluded folders (empty answers)
foldersWithEmptyAnswers = [
    "10_MPlus_ADAM_QA",
    "11_MPlusDrugs_QA",
    "12_MPlusHerbsSupplements_QA",
]

# Process XML files
BASE_PATH = "./MedQuAD-master"
for folder in os.listdir(BASE_PATH):
    if folder not in foldersWithEmptyAnswers:
        for xmlFileName in os.listdir(os.path.join(BASE_PATH, folder)):
            processXmlFile(completePath)
```

Key preprocessing steps:
- Extracts QA pairs and focus areas from XML
- Cleans and formats answer text
- Removes redundant whitespace and special characters
- Handles single and multiple QA pair cases
- Exports to CSV format (processed_medquad.csv)

### 2. Vector Store Builder (VectorBuilder.py)
- Processes the MedQuAD CSV data
- Creates document chunks with metadata
- Generates embeddings using BioBERT
- Builds and persists the ChromaDB vector store

```python
from src.VectorBuilder import VectorBuilder

builder = VectorBuilder("processed_medquad.csv")
documents = builder.prepare_documents()
chunks = builder.create_chunks(documents)
vectorstore = builder.create_vectorstore(chunks)
```

### 2. Question Retriever (SimilarQuestionRetriever.py)
- Custom retriever for finding similar medical questions
- Uses semantic similarity search
- Filters results based on question metadata
- Returns top matching documents

### 3. RAG System (MedicalRAGSystem.py)
- Core system integrating all components
- Uses BioMistral-7B for response generation
- Implements custom medical prompt template
- Provides source attribution for answers

```python
from src.MedicalRAGSystem import MedicalRAGSystem

rag_system = MedicalRAGSystem(
    model_path="BioMistral/BioMistral-7B",
    persist_directory="medical_vectorstore"
)
response = rag_system.get_answer("What are the symptoms of diabetes?")
```

### 4. Evaluation System (evaluator.py)
- Implements RAGAS evaluation metrics
- Uses Gemini Pro for evaluation
- Measures faithfulness, relevancy, and context quality
- Supports custom sample sizes

```python
from src.evaluator import RAGEvaluator

evaluator = RAGEvaluator(rag_system)
results = evaluator.run_evaluation(num_samples=5)
```

## Model Configuration

The system uses the following key configurations:

- **BioMistral Settings**:
  - Max length: 512 tokens
  - Temperature: 0.1
  - Top-p: 0.95
  - Repetition penalty: 1.1

- **Vector Store Settings**:
  - Chunk size: 500 characters
  - Chunk overlap: 50 characters
  - BioBERT embeddings with GPU acceleration when available

## Evaluation Metrics

The system is evaluated using RAGAS metrics:
- Faithfulness: Measures response accuracy
- Answer Relevancy: Assesses response appropriateness
- Context Recall: Evaluates information retrieval
- Context Precision: Measures context relevance

## Limitations

- Requires significant computational resources for optimal performance
- Model responses limited to MedQuAD dataset scope
- GPU recommended for reasonable inference speed
- Requires Google API key for evaluation system

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NIH websites for the original medical content
- BioMistral team for the biomedical language model
- DMIS Lab for BioBERT
- The LangChain community
- Kaggle for providing GPU resources

## Disclaimer

This system is for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
