# Biomedical RAG Chatbot

A biomedical question-answering system built on the MedQuAD dataset using RAG (Retrieval-Augmented Generation) with BioMistral and BioBERT embeddings. This system provides accurate medical information by leveraging a large dataset of medical Q&A pairs from trusted NIH sources.

## Key Achievements
- Processed and structured 47,457+ medical QA pairs from 12 NIH sources
- Successfully handles complex medical queries across multiple conditions
- Developed comprehensive evaluation pipeline with RAGAS metrics
- Answer Relevancy 91% and Faithfullness of 76%

## Overview

This project implements a medical domain question-answering system using the following components:

- **MedQuAD Dataset**: A comprehensive medical Q&A dataset containing 47,457 pairs from NIH websites
- **BioMistral-7B**: Large Language Model fine-tuned for biomedical text generation
- **BioBERT**: Domain-specific BERT model used for generating embeddings
- **ChromaDB**: Vector store for efficient similarity search
- **LangChain**: Framework for building the RAG pipeline

## System Architecture
```
[User Query] → [BioMistral-7B] → [BioBERT Embeddings] → [ChromaDB] → [Response Generation]
              ↑                                                      ↑
              └──────────────────[Context Retrieval]────────────────┘
```

## Development Environment

The project can be run in two environments:

1. **Local Environment**
   - Suitable for development and testing
   - Requires sufficient RAM and storage
   - GPU recommended for reasonable inference speed
   
2. **Kaggle Notebooks (Recommended)**
   - Recommended for speed, making vector and inference
   - Provides free GPU access (T4 x2) for 30/hours a week
   - Better performance for compute-intensive tasks

## Working with Notebooks

The project includes two main notebooks in the `notebooks/` directory:

### 1. medquadpreprocessing.ipynb 
- Handles initial XML data processing and CSV generation
- Removes three folders from MedQuAD having empty answers due to copyright
- Creates cleaned and structured dataset
- Generates processed_medquad.csv

### 2. biomed-chatbot-rag.ipynb
- Requires processed_medquad.csv as input in Kaggle
- Main RAG system implementation
- Handles model loading and vector store creation
- Includes evaluation system setup
  
To use notebooks on Kaggle:
1. Upload notebooks to a new Kaggle kernel
2. Select GPU accelerator (T4 x2)
3. Enable internet access
4. Run cells sequentially

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

### 3. Question Retriever (SimilarQuestionRetriever.py)
- Custom retriever for finding similar medical questions
- Uses semantic similarity search
- Filters results based on question metadata
- Returns top matching documents

### 4. RAG System (MedicalRAGSystem.py)
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

### 5. Evaluation System (evaluator.py)
- Implements RAGAS evaluation metrics
- Uses Gemini Pro for evaluation
- Measures faithfulness, relevancy, and context quality
- Supports custom sample sizes

```python
from src.evaluator import RAGEvaluator

evaluator = RAGEvaluator(rag_system)
results = evaluator.run_evaluation(num_samples=5)
```

Note: For evaluation metrics, the system uses Gemini Pro by default. If you have access to Gemini Pro 1.5 tokens, you can configure the evaluator to use the improved model:

```python
evaluator = RAGEvaluator(
    rag_system,
    model="gemini-pro-1.5",  # If you have access to Gemini-1.5 tokens
    api_key="YOUR_API_KEY"
)
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

## Real-World Examples

The following examples demonstrate the system's capabilities across different types of medical queries:

### 1. Diagnostic Query
```python
question = "How to diagnose Lung Cancer?"
response = rag_system.get_answer(question)

# Response:
"""
To diagnose lung cancer, a bronchoscopy is performed where a tube is inserted 
through the nose or mouth into the lungs to examine the lining of the bronchial 
tubes. A biopsy may also be taken during the procedure to check for cancer cells. 
Other diagnostic tests such as a CT scan, PET scan, and MRI may also be used to 
determine if the cancer has spread.

Source Focus: Lung Cancer
"""
```

### 2. Symptom Assessment Query
```python
question = "What are the symptoms of Parkinson's Disease?"
response = rag_system.get_answer(question)

# Response:
"""
The symptoms of Parkinson's disease include tremors, rigidity, bradykinesia 
(slow movement), and postural instability. Other common symptoms include 
difficulty with balance and walking, fatigue, sleep disturbances, and 
constipation.

Source Focus: Parkinson's Disease
"""
```

### 3. Treatment Query
```python
question = "What are the treatments for Type 1 Diabetes?"
response = rag_system.get_answer(question)

# Response:
"""
The treatment for type 1 diabetes involves insulin replacement therapy and 
self-monitoring of blood glucose levels. Insulin replacement therapy can be 
achieved through multiple daily injections or continuous subcutaneous insulin 
infusion. It is also important for individuals with type 1 diabetes to monitor 
their blood glucose levels regularly and adjust their insulin doses accordingly.

Source Focus: Type 1 Diabetes
"""
```

### 4. Complex Medical Relationship Query
```python
question = "How does diabetes affect the kidneys and heart?"
response = rag_system.get_answer(question)

# Response:
"""
Diabetes can cause damage to the small blood vessels in the body, including 
the heart and kidneys. High blood sugar levels can lead to heart disease, 
stroke, and kidney disease. The risk for these complications increases as 
blood sugar levels get higher and stay high over time.

Source Focus: Diabetes Complications
"""
```

## Evaluation Metrics

The system is evaluated using RAGAS metrics:
- Faithfulness: Measures response accuracy
- Answer Relevancy: Assesses response appropriateness
- Context Recall: Evaluates information retrieval
- Context Precision: Measures context relevance

## Challenges Overcome
- Optimized large-scale medical data processing pipeline
- Built efficient semantic search for medical queries
- Handled complex medical terminology and relationships
- Balanced response accuracy with generation speed
- Developed comprehensive evaluation metrics

## Future Enhancements
- Integration with additional medical databases
- Real-time medical literature updates
- Multi-language support
- Enhanced context understanding for rare diseases
- Improved evaluation metrics using newer LLMs (eg Gemini Pro 1.5)
- Utilize more API tokens

## Limitations

- Requires significant computational resources for optimal performance
- Model responses limited to MedQuAD dataset scope
- GPU recommended for reasonable inference speed
- Requires Google API key for evaluation system
- Very limited API tokens and not using most recent Gemini Pro Model

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
