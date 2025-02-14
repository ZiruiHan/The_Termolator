from sentence_transformers import SentenceTransformer
import numpy as np

reference_terms = [
    "computer vision",
]

# Load candidate terms from file
input_file = "/Users/ziruihan/Desktop/independent_termolator/term_result/Computer_vision_2000.out_term_list"
with open(input_file, 'r', encoding='utf-8') as f:
    candidate_terms = [line.strip() for line in f if line.strip()]

# Load a sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode terms
reference_embeddings = model.encode(reference_terms, convert_to_numpy=True)
candidate_embeddings = model.encode(candidate_terms, convert_to_numpy=True)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarity_scores = []

for i, candidate_emb in enumerate(candidate_embeddings):
    # Compare candidate term embedding to each reference embedding
    similarities = [cosine_similarity(candidate_emb, ref_emb) for ref_emb in reference_embeddings]
    max_similarity = max(similarities)
    candidate_term = candidate_terms[i]
    similarity_scores.append(max_similarity)
    print(f"Term: '{candidate_term}' | Max Similarity to AI terms: {max_similarity:.4f}")

# Compute the average similarity score
if similarity_scores:
    average_similarity = np.mean(similarity_scores)
    print(f"\nAverage Similarity for all candidate terms: {average_similarity:.4f}")
else:
    print("No candidate terms found.")
