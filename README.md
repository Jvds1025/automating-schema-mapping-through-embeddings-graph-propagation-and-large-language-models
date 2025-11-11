# Automating schema mapping through embedding, large language models and graph propagation
This repository contains the code developed for the thesis “Automating Schema Mapping through Embeddings, Large Language Models, and Graph Propagation.” The project investigates automated methods to map source schema columns to target columns using LoRA fine-tuned embeddings, graph propagation, and prompt-based LLM inference.

**Key Features**
* Embedding Fine-Tuning: Apply LoRA to the Qwen3-Embedding-4B model for parameter-efficient adaptation. The model is trained to minimize cosine similarity loss between paired schema embeddings, allowing similar columns to be closer in the embedding space while dissimilar columns are pushed apart.

* Candidate Generation: Compute pairwise cosine similarity between source and target embeddings to generate potential mappings above a defined threshold.

* Decoding Strategies:
  * Graph Propagation: Constructs a graph from embeddings to capture one-to-many relationships and structural dependencies across tables. Post-processing with human-in-the-loop for ambiguous mappings is applied only for graph-based decoding.
  * Prompt-Based Few-Shot Inference: Uses a language model to predict mappings for unseen schema pairs.
* Evaluation: Supports accuracy, precision, recall, and F1 scores for top-1 and top-3 candidate selections.
* Utilities: Thresholding, visualization of embeddings and mappings, and optional human validation.

**Notes**
* Original datasets are confidential due to partner restrictions. Scripts include synthetic examples to reproduce workflows.
* The repository enables reproduction of the full pipeline, including embedding fine-tuning, candidate mapping generation, graph-based and LLM decoding, and evaluation metrics.


This work demonstrates how modern embedding methods, combined with graph-based propagation and LLMs, can improve automated schema mapping. The approach balances semantic accuracy, structural awareness, and computational efficiency, providing a practical framework for large-scale heterogeneous data integration tasks.
