# CA Exam Retrieval-Based QA Bot

This project implements a **retrieval-based question–answering (QA) bot** designed to assist with **Chartered Accountant (CA) exam preparation**.  
The system enables users to query large volumes of CA study material and retrieve **conceptually relevant explanations** using semantic vector search rather than simple keyword matching.

---

## Project Overview

CA exam preparation involves navigating extensive, concept-heavy documentation across multiple subjects. Traditional search methods are often inefficient, as they depend on exact keyword matches and fail to capture semantic intent.

This project addresses this challenge by building a **semantic retrieval pipeline** that:
- Converts CA study material into dense vector embeddings  
- Indexes the embeddings using efficient similarity search  
- Retrieves the most relevant content in response to natural language queries  

The result is a domain-specific study assistant capable of handling varied question phrasing while maintaining conceptual accuracy.

---

## Methodology

The system follows a **retrieval-augmented approach**:

1. CA study material (PDFs) is extracted and divided into semantically coherent text chunks  
2. Each chunk is transformed into a dense vector representation using a pre-trained embedding model  
3. The vectors are indexed using a similarity search index for fast retrieval  
4. User queries are embedded into the same vector space  
5. The most semantically similar content is retrieved and presented as the answer context  

This design decouples content storage from query understanding, making the system scalable and easy to extend with new material.

---

## Models Used

The system is composed of three main modeling components: an embedding model for semantic representation, a similarity search mechanism for retrieval, and a language model for answer generation.

### Embedding Model
A **pre-trained dense embedding model (BGE family)** is used to convert CA study material and user queries into fixed-length vector representations.  
Unlike keyword-based methods, these embeddings capture semantic meaning, allowing the system to retrieve relevant content even when the query phrasing differs from the source text.

The same embedding model is applied to both documents and queries to ensure consistency within a shared vector space.

### Similarity Search
To enable efficient retrieval, the document embeddings are indexed using **FAISS**, a high-performance similarity search library.  
Cosine similarity (implemented via normalized inner product search) is used to identify the most semantically similar text chunks for a given query. This allows fast and scalable retrieval even when the knowledge base grows large.

### Language Model
A **lightweight transformer-based instruction-following language model** is used to generate answers based on the retrieved content.  
The retrieved study material is provided as context to the language model, ensuring that responses are grounded in the source documents rather than generated purely from prior knowledge.

A fallback language model is included to ensure robustness in environments with limited computational resources.

### Design Rationale
By combining dense embeddings for semantic understanding, vector similarity search for retrieval, and a transformer-based language model for response generation, the system balances accuracy, interpretability, and efficiency. This modular design makes it well-suited for exam-focused educational applications such as CA preparation.

---

## Project Structure

