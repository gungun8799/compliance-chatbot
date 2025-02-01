import re
import json

def extract_faq(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    faqs = []
    category = None
    question = None
    answer = []

    def store_current_q():
        """Helper function to store the current question and answer if valid."""
        if question and answer and category:
            faqs.append({
                "category": category,
                "question": question,
                "answer": " ".join(answer)
            })

    for line in lines:
        line = line.strip()

        # Ignore blank lines, separator lines, or '* * *'
        if not line or re.match(r'[-]+$', line) or line == '* * *':
            continue

        # Check if it's a new category line
        if "(categories)" in line:
            # First, store the previous question under the old category
            store_current_q()
            # Set new category
            category = line.replace("(categories)", "").strip()
            # Reset question and answer so they don't carry over incorrectly
            question = None
            answer = []
            continue

        # Check if it's a new question
        if line.endswith('?'):
            # Store the previous question if exists
            store_current_q()
            # Start a new question
            question = line
            answer = []
            continue

        # Otherwise, this line is part of the answer
        answer.append(line)

    # After the loop, store the last question if present
    store_current_q()

    return faqs

# Example usage
if __name__ == "__main__":
    file_path = "walmart_faqs.txt"  # Adjust path as needed
    faqs = extract_faq(file_path)

    # Save as JSON
    with open("walmart_faqs.json", "w", encoding="utf-8") as json_file:
        json.dump(faqs, json_file, indent=4, ensure_ascii=False)

    # Print extracted FAQs (preview)
    for faq in faqs[:5]:
        print(f"Category: {faq['category']}")
        print(f"Question: {faq['question']}")
        print(f"Answer: {faq['answer']}")
        print("-" * 80)

    # Summary Report
    num_categories = len(set(faq['category'] for faq in faqs if faq['category']))
    num_questions = len(faqs)
    print("\nSummary Report")
    print("=" * 40)
    print(f"Total Categories: {num_categories}")
    print(f"Total Questions: {num_questions}")
