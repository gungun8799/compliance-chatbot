# token_utils.py

import tiktoken
from prompts import SYSTEM_PROMPT_ORIGINAL, SYSTEM_PROMPT_310125

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count the number of tokens in the given text using the specified model's encoding.
    
    Parameters:
        text (str): The text to tokenize.
        model (str): The model name to determine the encoding. Defaults to "gpt-4".
    
    Returns:
        int: The number of tokens in the text.
    """
    try:
        # Try to get the encoding for the specified model.
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback encoding if the model is not recognized.
        encoding = tiktoken.get_encoding("cl100k_base")
    
    # Encode the text and return the token count.
    tokens = encoding.encode(text)
    return len(tokens)

if __name__ == "__main__":
    token_count_original = count_tokens(SYSTEM_PROMPT_ORIGINAL)
    token_count_310125 = count_tokens(SYSTEM_PROMPT_310125)
    print("token_count_original:", token_count_original)
    print("token_count_310125:", token_count_310125)