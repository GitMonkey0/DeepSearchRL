import re
import json
from typing import List, Union, Dict
import random

# =================================================================
# F1 Score Calculation Helpers
# =================================================================

def normalize(text: str) -> List[str]:
    """
    Lowercase, tokenize, and remove punctuation from text to get a bag of words.
    """
    return re.findall(r'\b\w+\b', text.lower())

def f1_single(pred: str, gold: str) -> float:
    """
    Calculates the F1 score for a single prediction and a single ground truth answer.
    """
    pred_tokens = set(normalize(pred))
    gold_tokens = set(normalize(gold))
    
    # Handle edge cases where one or both are empty
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
        
    common_tokens = pred_tokens & gold_tokens
    if not common_tokens:
        return 0.0
        
    precision = len(common_tokens) / len(pred_tokens)
    recall    = len(common_tokens) / len(gold_tokens)
    
    return 2 * precision * recall / (precision + recall)

# =================================================================
# Solution String Parsing and Validation Helpers
# =================================================================

def get_tool_call_contents(s: str) -> List[str]:
    """
    Extracts all contents wrapped in <tool_call>...</tool_call> tags.
    """
    return re.findall(r"<tool_call>(.*?)</tool_call>", s, flags=re.DOTALL)

def extract_solution(solution_str: str) -> Union[str, None]:
    """
    Extracts the content from the last <answer>...</answer> tag pair.
    """
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()

def is_legal(solution_str: str) -> bool:
    """
    Performs a unified legality check on the solution string.
    
    This function checks for:
    1.  Correctly closed and paired <tool_call> tags.
    2.  Valid JSON structure within each tool call.
    3.  No duplicate tool calls.
    4.  Exactly one pair of <answer> tags with extractable content.
    
    Returns False for any violation, otherwise True.
    """
    # ---- 1. Check for unclosed or mismatched tool_call tags ----
    if solution_str.count("<tool_call>") != solution_str.count("</tool_call>"):
        return False

    tool_contents = get_tool_call_contents(solution_str)
    
    # ---- 2. Check for duplicate tool calls ----
    normalized_contents = [c.replace("\n", "").strip() for c in tool_contents]
    if len(normalized_contents) != len(set(normalized_contents)):
        return False
        
    # ---- 3. Check JSON validity and structure of each tool call ----
    for content in tool_contents:
        try:
            data = json.loads(content.strip())
            # Validate the required keys and their types
            if not (
                isinstance(data, dict) and
                "name" in data and isinstance(data["name"], str) and
                "arguments" in data and isinstance(data["arguments"], dict) and
                set(data["arguments"].keys()) == {"query"} and
                isinstance(data["arguments"]["query"], str)
            ):
                return False
        except json.JSONDecodeError:
            return False # Malformed JSON is illegal

    # ---- 4. Check for exactly one valid answer tag ----
    if solution_str.count("<answer>") != 1 or solution_str.count("</answer>") != 1:
        return False
    
    if extract_solution(solution_str) is None:
        return False
        
    # If all checks pass, the solution is deemed legal
    return True

# =================================================================
# Main Scoring Function (Refactored)
# =================================================================

def compute_score(solution_str: str, ground_truth: Dict) -> float:
    """
    Computes the final score for a given solution string based on legality and F1 score.
    - Returns -1.0 for any illegally formatted solution string.
    - Otherwise, returns the maximum F1 score between the extracted answer
      and the list of possible golden answers.
    """
    # First, validate the entire format of the solution string.
    do_print = random.randint(1, 64) == 1
    predicted_answer = extract_solution(solution_str)

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        if predicted_answer is not None:
            print(f"Extracted answer is not None: {predicted_answer}")
        else:
            print("Extracted answer: None!")
        print(f"Solution string: {solution_str}")

    if not is_legal(solution_str):
        return {"score": -1.0, "acc": 0.0}
                
    # Ensure the golden answers from the ground truth are in a list.
    golden_answers = ground_truth.get("target")
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    if not golden_answers:
        # If there are no golden answers, score is 1.0 if prediction is also empty, else 0.0.
        score = 1.0 if not predicted_answer.strip() else 0.0
        return {"score": score, "acc": score}

    # Calculate F1 against each possible golden answer and take the highest score.
    max_f1 = max(f1_single(predicted_answer, gold) for gold in golden_answers)
    return {"score": max_f1, "acc": max_f1}
