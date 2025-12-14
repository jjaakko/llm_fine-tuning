import re


def extract_prediction(item: str) -> str:
    """Extract predicted account number from the llm output.

    Args:
        item (str): LLM output.

    Returns:
        str: account number. If multiple sequences match the pattern returns the last one.
    """
    try:
        # Find all digit sequences
        all_digit_sequences = re.findall(r"\d+", item)

        # Filter for valid sequences (4-7 digits) that are not part of longer sequences
        valid_sequences = []
        for seq in all_digit_sequences:
            if 4 <= len(seq) <= 7:
                valid_sequences.append(seq)

        # Return the last valid sequence if any exist.
        if valid_sequences:
            return valid_sequences[-1]
    except Exception as e:
        # Valid prediction not found.
        print("Error extracting prediction:")
        print(item)
        print(e)

    return "-1"
