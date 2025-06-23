from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch

class EmbeddingModel:
    def __init__(self, top_k=2):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.top_k = top_k
    
    def query(self, query, text_list):
        # Ensure text_list is a list
        if isinstance(text_list, str):
            text_list = [text_list]
            
        # Convert to numpy arrays first
        chunk_embeddings = self.model.encode(text_list, convert_to_tensor=False)
        query_embedding = self.model.encode([query], convert_to_tensor=False)
        
        # Normalize the embeddings
        chunk_embeddings = self.normalize(chunk_embeddings)
        query_embedding = self.normalize(query_embedding)
        
        # Calculate cosine similarity
        cosine_similarities = self.cosine_similarity(chunk_embeddings, query_embedding).squeeze()
        
        # Handle single text case
        if len(text_list) == 1:
            return [{
                'text': text_list[0],
                'score': float(cosine_similarities)
            }]
        
        # Get top k results for multiple texts
        top_indices = np.argsort(-cosine_similarities)[:self.top_k]
        
        # Get texts and their scores
        results = []
        for idx in top_indices:
            results.append({
                'text': text_list[idx],
                'score': float(cosine_similarities[idx])
            })
        
        return results
    
    def normalize(self, vecs):
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    def encode(self, sentences):
        return self.model.encode(sentences, convert_to_tensor=False)

    def cosine_similarity(self, embeddings1, embeddings2):
        # Convert to torch tensors
        embeddings1 = torch.tensor(embeddings1)
        embeddings2 = torch.tensor(embeddings2)
        
        # Calculate cosine similarity
        similarity = util.cos_sim(embeddings1, embeddings2)
        
        # Move to CPU and convert to numpy
        return similarity.cpu().numpy()

if __name__ == "__main__":
    model = EmbeddingModel(top_k=2)
    
    # Example texts about different scientific topics
    texts = [
        "Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles. It is the foundation of all quantum physics including quantum chemistry, quantum field theory, quantum technology, and quantum information science.",
        
        "The theory of relativity is a theory in physics that describes the relationship between space and time. It was first proposed by Albert Einstein in 1905 and is based on the principle of relativity, which states that the laws of physics are the same for all observers in uniform motion.",
        
        "Evolution is the process by which different kinds of living organisms developed from earlier forms during the history of the Earth. The theory of evolution by natural selection was first formulated in Darwin's book 'On the Origin of Species' in 1859.",
        
        "Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, but since the 1800s, human activities have been the main driver of climate change, primarily due to burning fossil fuels like coal, oil and gas.",
        
        "Artificial Intelligence (AI) is the simulation of human intelligence by machines that are programmed to think and learn like humans. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving."
    ]
    
    # Example queries
    queries = [
        "What is quantum physics?",
        "Tell me about Einstein's theory",
        "How do species evolve?",
        "What causes global warming?",
        "What is machine learning?"
    ]
    
    # Test each query
    for query in queries:
        print(f"\nQuery: {query}")
        print("Most relevant texts:")
        results = model.query(query, texts)
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result['score']:.4f}")
            print(f"   Text: {result['text'][:150]}...")  # Print first 150 characters of each result