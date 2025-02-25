Problem Statement
The objective of this project is to develop an AI-powered compliance agent that helps in analyzing and understanding NDMO regulations using Retrieval-Augmented Generation (RAG) techniques and large language models (LLMs). The system enables users to accurately retrieve legal and regulatory information, facilitating compliance with various regulations.

Dataset Description
The dataset is based on regulatory documents issued by the National Data Management Office (NDMO) and includes:

National data governance policies.
Personal data protection and privacy regulations.
Compliance guidelines for AI usage.
Laws related to cross-border data transfers.
Tools & Technologies
Libraries & Frameworks
pandas – Data analysis and processing.
NumPy – Numerical and statistical computations.
Hugging Face Transformers – Pre-trained AI models for NLP.
LangChain – RAG framework for legal information retrieval and response generation.
ChromaDB – Vector database for efficient legal document retrieval.
Gradio – Interactive interface for compliance-related queries.
Algorithms
LLaMA 2-7B Fine-Tuned – Optimized model for regulatory and legal queries.
Retrieval-Augmented Generation (RAG) – For generating responses based on regulatory documents.
LoRA (Low-Rank Adaptation) – Enhances model efficiency with minimal resource consumption.
4-bit Quantization – Reduces memory usage and improves inference efficiency.
Model Evaluation Metrics
The model has been evaluated using the following metrics:

BLEU Score – Measures similarity between generated responses and reference texts.
ROUGE Scores – Evaluates accuracy and coverage of legal responses:
ROUGE-1
ROUGE-2
ROUGE-L
BERTScore – Assesses response accuracy based on contextual understanding.
