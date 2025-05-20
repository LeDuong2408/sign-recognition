import ast


def parse_str_2_list(input: str) -> list[str]:
    try:
      if not input:
          print("[ERROR POST OCR] Input is empty")
          return []
      return ast.literal_eval(input)
    except (ValueError, SyntaxError):
      print("[ERROR POST OCR] Error parsing string to list")
      return []