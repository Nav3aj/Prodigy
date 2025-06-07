"""generate_markov.py
Instruction: Generate text using a Markov chain of specified order.
"""

# Topic: Imports
import random
import argparse

# Topic: Building the Markov Chain model
def build_markov(text, order=1):
    """Builds a Markov chain model (order-n) from the input text."""
    chains = {}
    tokens = text.split()
    for i in range(len(tokens) - order):
        key = tuple(tokens[i:i+order])
        next_word = tokens[i+order]
        chains.setdefault(key, []).append(next_word)
    return chains

# Topic: Generating text from the model
def generate_text(chains, order, length=50):
    """Generates text of 'length' words based on the Markov chains."""
    # Start with a random state
    state = random.choice(list(chains.keys()))
    output = list(state)
    for _ in range(length - order):
        next_words = chains.get(state)
        if not next_words:
            break
        word = random.choice(next_words)
        output.append(word)
        state = tuple(output[-order:])
    return ' '.join(output)

# Topic: Command-line interface
def main(input_path, order, length):
    # Load corpus
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # Build and generate
    chains = build_markov(text, order)
    result = generate_text(chains, order, length)
    print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Markov chain text generator")
    parser.add_argument("--input", required=True, help="Path to input text file")
    parser.add_argument("--order", type=int, default=2, help="Order of the Markov model")
    parser.add_argument("--length", type=int, default=100, help="Number of words to generate")
    args = parser.parse_args()
    main(args.input, args.order, args.length)
