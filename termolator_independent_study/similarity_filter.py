from sentence_transformers import SentenceTransformer
import numpy as np

# Paths to your files
domain_terms_file = "/Users/ziruihan/Desktop/independent_termolator/computer_vision_answerkey.txt"
candidate_terms_file = "/Users/ziruihan/Desktop/independent_termolator/term_result/Computer_vision_2000.out_term_list"
output_file = "/Users/ziruihan/Desktop/independent_termolator/Computer_vision_200.filtered_and_ranked.txt"
# Define your threshold
threshold = 0.5  # Adjust this value as needed

# Load the domain-specific (seed) terms
with open(domain_terms_file, 'r', encoding='utf-8') as f:
    domain_terms = [line.strip() for line in f if line.strip()]

# Load the candidate terms from the Termolator output
with open(candidate_terms_file, 'r', encoding='utf-8') as f:
    candidate_terms = [line.strip() for line in f if line.strip()]

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute embeddings
domain_embeddings = model.encode(domain_terms, convert_to_numpy=True)
candidate_embeddings = model.encode(candidate_terms, convert_to_numpy=True)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Compute max similarity for each candidate term
results = []
for candidate_term, candidate_emb in zip(candidate_terms, candidate_embeddings):
    # Compute similarity to each domain term and take max
    similarities = [cosine_similarity(candidate_emb, d_emb) for d_emb in domain_embeddings]
    max_similarity = max(similarities) if similarities else 0.0
    results.append((candidate_term, max_similarity))

# Sort results by similarity in descending order
results.sort(key=lambda x: x[1], reverse=True)

# Apply the threshold to filter out less relevant terms
filtered_results = [(term, score) for term, score in results if score >= threshold]


print("Top 10 candidate terms (after thresholding) by semantic similarity to domain:")
for term, score in filtered_results[:10]:
    print(f"Term: {term}\tScore: {score:.4f}")

# Write all filtered results to a file
with open(output_file, 'w', encoding='utf-8') as out:
    out.write("Term\tMaxSimilarity\n")
    for term, score in filtered_results:
        out.write(f"{term}\t{score:.4f}\n")

print(f"\nFiltered and ranked terms (above {threshold}) saved to: {output_file}")
