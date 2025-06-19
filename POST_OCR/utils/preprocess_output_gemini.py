import json


def decode_unicode_if_needed(text: str) -> str:
    try:
        return (
            bytes(text, "utf-8").decode("unicode_escape")
            if r"\\u" in text or r"\\U" in text or r"\\ U" in text
            else text
        )
    except Exception as e:
        print(f"Error decoding unicode: {e}")
        return text
