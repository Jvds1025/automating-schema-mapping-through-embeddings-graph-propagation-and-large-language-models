# Automating Schema Mapping Through Large Language Models, and Graph Propagation

This repository contains the code developed for the thesis **“Automating Schema Mapping through Large Language Models, and Graph Propagation.”** The project investigates automated approaches for mapping source schema columns to target columns using LoRA fine-tuned embeddings, graph propagation, and generative LLM prompting.  

## Key Features

* **Embedding Fine-Tuning:**  
  Apply LoRA to the **Qwen3-Embedding-4B** model for parameter-efficient adaptation. The embeddings are fine-tuned to capture semantic similarity between schema columns, enabling the model to distinguish mappable and non-mappable columns.  

* **Candidate Generation:**  
  Compute pairwise cosine similarity between source and target embeddings to generate candidate mappings. Thresholding ensures only high-confidence candidates are considered.  

* **Added models:**  
  * **Graph Propagation:**  
    Refines column-level similarities by leveraging global table-level relationships, reducing false positives and improving identification of non-mappable columns.  
  * **Generative LLM Prompting:**  
    Uses **Qwen3-4B-Instruct 2507** to resolve ambiguous mappings and identify non-mappable columns through instruction-based few-shot inference.  
  * **Human-in-the-Loop Benchmark:**  
    Provides a semi-automatic configuration to evaluate maximal achievable accuracy and serves as a performance benchmark for automated methods.  

* **Evaluation Metrics:**  
  Supports **accuracy, precision, recall, and F1-score** for both top-1 and top-3 candidate predictions. Efficiency metrics are runtime, token-usage and peak memory

* **Utilities:**  
  Includes thresholding, similarity propagation and candidate ranking

## Notes

* The repository enables reproduction of the full pipeline: embedding fine-tuning, candidate generation, graph-based similarity refinement, LLM-based inference, and evaluation metrics.  

---

This implementation demonstrates how **combining embeddings, structural reasoning via graph propagation, and generative LLM inference** improves automated schema mapping. The approach balances semantic accuracy, structural awareness, and computational efficiency, providing a practical framework for large-scale heterogeneous data integration.
