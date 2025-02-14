import os
import csv

def load_term_list(file_path):
    """Load all terms and their variants from a term list file."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return set()
    term_list = set()
    with open(file_path, 'r') as file:
        for line in file:
            terms = line.strip().lower().split('\t')  # Parse and split variants by tab
            term_list.update(terms)
    return term_list

def load_manual_glossary(csv_path):
    """Load the manually labeled glossary (100 items) from a CSV file."""
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return set(), set()
    true_terms = set()
    false_terms = set()
    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            term = row['Term'].strip().lower()
            if row['Label'].strip().lower() == 'term':
                true_terms.add(term)
            else:
                false_terms.add(term)
    return true_terms, false_terms

def calculate_precision(system_terms, true_terms, false_terms):
    """Estimate precision based on the manually labeled sample."""
    manual_sample_terms = true_terms.union(false_terms)  # All 100 manually labeled terms
    true_positives = system_terms.intersection(true_terms)
    precision = len(true_positives) / len(manual_sample_terms) if manual_sample_terms else 0.0
    print(f"\nPrecision Estimate: {precision:.2%} ({len(true_positives)} correct out of {len(manual_sample_terms)} manual terms)")
    return precision

def main():
    term_list_path = "/Users/ziruihan/Desktop/independent_termolator/term_result/Computer_vision_2000.out_term_list"
    manual_glossary_path = "/Users/ziruihan/Desktop/independent_termolator/labeled_term/Labeled_Terms_2000.csv"
    print("\nLoading term list...")
    system_terms = load_term_list(term_list_path)

    print("Loading manual glossary...")
    true_terms, false_terms = load_manual_glossary(manual_glossary_path)

    print("\nCalculating precision and recall...")
    precision = calculate_precision(system_terms, true_terms, false_terms)

    print("\nEvaluation Results:")
    print(f"Precision Estimate: {precision:.2%}")

if __name__ == "__main__":
    main()


