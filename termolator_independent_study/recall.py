import os
import re
from difflib import SequenceMatcher

def normalize_term(term):
    """Normalize a term to handle variations like pluralization and whitespace."""
    term = term.lower().strip()  # Convert to lowercase and remove leading/trailing spaces
    term = re.sub(r'\s+', ' ', term)  # Replace multiple spaces with a single space
    term = re.sub(r'[^a-z0-9 ]', '', term)  # Remove special characters
    if term.endswith('s'):  # Handle simple pluralization
        term = term[:-1]
    return term

def load_terms(file_path):
    """Load terms from a file into a normalized set."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return set()
    with open(file_path, 'r') as file:
        return {normalize_term(line) for line in file}

def is_fuzzy_match(term1, term2, threshold=0.85):
    """Check if two terms are similar enough based on a threshold."""
    return SequenceMatcher(None, term1, term2).ratio() >= threshold

def evaluate_with_answer_key(system_terms, answer_key_terms):
    """Evaluate precision and recall using the answer key with fuzzy matching."""
    true_positives = set()

    for answer_key_term in answer_key_terms:
        normalized_answer_key = normalize_term(answer_key_term)
        for system_term in system_terms:
            normalized_system_term = normalize_term(system_term)
            if is_fuzzy_match(normalized_answer_key, normalized_system_term):
                true_positives.add(answer_key_term)
                break

    # Precision: Fraction of system terms that match the answer key
    precision = len(true_positives) / len(system_terms) if system_terms else 0.0

    # Recall: Fraction of answer key terms that match the system terms
    recall = len(true_positives) / len(answer_key_terms) if answer_key_terms else 0.0

    print(f"\nEvaluation Results:")
    print(f"Precision: {precision:.2%} ({len(true_positives)} matched out of {len(system_terms)} system terms)")
    print(f"Recall: {recall:.2%} ({len(true_positives)} matched out of {len(answer_key_terms)} answer key terms)")

    return precision, recall


def main():
    term_list_path = "/Users/ziruihan/Desktop/independent_termolator/cv_new_terms.txt"
    answer_key_path = "/Users/ziruihan/Desktop/independent_termolator/computer_vision_answerkey.txt"

    print("\nLoading term list...")
    system_terms = load_terms(term_list_path)

    print("Loading answer key...")
    answer_key_terms = load_terms(answer_key_path)

    print("\nEvaluating against the answer key with fuzzy matching...")
    precision, recall = evaluate_with_answer_key(system_terms, answer_key_terms)

if __name__ == "__main__":
    main()
