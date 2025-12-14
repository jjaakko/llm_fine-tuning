from finetune.text_extraction import extract_prediction


class TestExtractFunctions:
    def test_extract_prediction_with_valid_input(self):
        # Test with a string containing a valid pattern (last 4-7 digits)
        assert extract_prediction("This is a test 1234 and some more text") == "1234"
        assert extract_prediction("Multiple numbers 5678 and 9012") == "9012"
        assert extract_prediction("Longer number 1234567") == "1234567"
        # Valid sequence with another valid sequence elsewhere
        assert extract_prediction("First 1234 and then 5678") == "5678"

    def test_extract_prediction_with_no_match(self):
        # Test with strings that don't match the pattern
        assert extract_prediction("No digits here") == "-1"
        assert extract_prediction("Only three 123") == "-1"
        # Longer sequences should not produce partial matches
        assert extract_prediction("Too many 12345678") == "-1"
        # Valid sequence followed by invalid longer sequence
        assert extract_prediction("Valid 1234 and invalid 12345678") == "1234"

    def test_extract_prediction_with_edge_cases(self):
        # Test edge cases
        assert extract_prediction("") == "-1"
        # The function doesn't match 3-digit numbers
        assert extract_prediction("123 456 789") == "-1"
        assert extract_prediction("1234 in the middle 5678 end") == "5678"
        # Multiple valid sequences with a longer invalid one
        assert extract_prediction("Valid 1234 and 5678 but not 123456789") == "5678"
