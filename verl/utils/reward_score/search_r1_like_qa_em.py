import re
import json
from typing import List, Union, Dict

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
    if not is_legal(solution_str):
        return -1.0

    # If legal, extract the predicted answer.
    predicted_answer = extract_solution(solution_str)
    
    # Ensure the golden answers from the ground truth are in a list.
    golden_answers = ground_truth.get("target")
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
        
    if not golden_answers:
        # If there are no golden answers, score is 1.0 if prediction is also empty, else 0.0.
        return 1.0 if not predicted_answer.strip() else 0.0

    # Calculate F1 against each possible golden answer and take the highest score.
    max_f1 = max(f1_single(predicted_answer, gold) for gold in golden_answers)
    
    return max_f1


# import random
# import re
# import string
# import json


# def is_valid_tool_call_string(s: str) -> bool:
#     matches = re.findall(r"<tool_call>(.*?)</tool_call>", s, flags=re.DOTALL)

#     if not matches and ("<tool_call>" in s or "</tool_call>" in s):
#         return False

#     for content in matches:
#         content = content.replace("\n", "").strip()

#         try:
#             data = json.loads(content)
#         except Exception:
#             return False  

#         if not isinstance(data, dict):
#             return False
#         if "name" not in data or "arguments" not in data:
#             return False
#         if not isinstance(data["name"], str):
#             return False

#         arguments = data["arguments"]
#         if not isinstance(arguments, dict):
#             return False
#         if set(arguments.keys()) != {"query"}:
#             return False

#         query = arguments["query"]
#         if not isinstance(query, str):
#             return False

#     return True


# def has_unclosed_tool_call(s: str) -> bool:
#     """检测是否存在不闭合的tool_call标签"""
#     return ("<tool_call>" in s or "</tool_call>" in s) and not re.findall(r"<tool_call>(.*?)</tool_call>", s, flags=re.DOTALL)


# def normalize_answer(s):
#     def remove_articles(text):
#         return re.sub(r"\b(a|an|the)\b", " ", text)

#     def white_space_fix(text):
#         return " ".join(text.split())

#     def remove_punc(text):
#         exclude = set(string.punctuation)
#         return "".join(ch for ch in text if ch not in exclude)

#     def lower(text):
#         return text.lower()

#     return white_space_fix(remove_articles(remove_punc(lower(s))))


# def em_check(prediction, golden_answers):
#     if isinstance(golden_answers, str):
#         golden_answers = [golden_answers]
#     normalized_prediction = normalize_answer(prediction)
#     score = 0
#     for golden_answer in golden_answers:
#         golden_answer = normalize_answer(golden_answer)
#         if golden_answer == normalized_prediction:
#             score = 1
#             break
#     return score


# def subem_check(prediction, golden_answers):
#     if isinstance(golden_answers, str):
#         golden_answers = [golden_answers]
#     normalized_prediction = normalize_answer(prediction)
#     score = 0
#     for golden_answer in golden_answers:
#         golden_answer = normalize_answer(golden_answer)
#         if golden_answer in normalized_prediction:
#             score = 1
#             break
#     return score


# def extract_solution(solution_str):
#     """Extract the answer from the solution string."""
#     answer_pattern = r"<answer>(.*?)</answer>"
#     match = re.finditer(answer_pattern, solution_str, re.DOTALL)
#     matches = list(match)

#     if len(matches) < 1:
#         return None
#     return matches[-1].group(1).strip()


# def count_answer_tags(text):
#     opening_tags = text.count("<answer>")
#     closing_tags = text.count("</answer>")
#     return opening_tags, closing_tags


# def get_tool_call_contents(s: str):
#     """提取所有 <tool_call>...</tool_call> 的内容"""
#     return re.findall(r"<tool_call>(.*?)</tool_call>", s, flags=re.DOTALL)


# def has_duplicate_tool_calls(s: str) -> bool:
#     """判断是否存在完全重复的 tool_call 内容"""
#     contents = get_tool_call_contents(s)
#     normalized = [c.replace("\n", "").strip() for c in contents]
#     return len(normalized) != len(set(normalized))


# # ========== 合法性检测 ==========

# def check_legality(solution_str: str) -> tuple[bool, bool]:
#     """统一的合法性检测：tool_call + answer
#     返回 (is_legal, has_unclosed_tool_call)
#     """
#     # ---- Tool call 检查 ----
#     has_unclosed = has_unclosed_tool_call(solution_str)
    
#     if not is_valid_tool_call_string(solution_str):
#         return False, has_unclosed
#     if has_duplicate_tool_calls(solution_str):
#         return False, has_unclosed

#     # ---- Answer 检查 ----
#     open_count, close_count = count_answer_tags(solution_str)
#     if open_count != 1 or close_count != 1:  # 必须严格只有一对
#         return False, has_unclosed

#     answer = extract_solution(solution_str)
#     if answer is None:
#         return False, has_unclosed

#     return True, has_unclosed


# # ========== 质量检测 ==========

# def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
#     """精确匹配 (EM) 评分"""
#     is_legal, has_unclosed = check_legality(solution_str)
    
#     # 对不闭合的tool_call给予-1的严厉惩罚
#     if has_unclosed:
#         return -1
    
#     if not is_legal:
#         return 0

#     answer = extract_solution(solution_str)
#     do_print = random.randint(1, 64) == 1
#     if do_print:
#         print("--------------------------------")
#         print(f"Golden answers: {ground_truth['target']}")
#         print(f"Extracted answer: {answer}")
#         print(f"Solution string: {solution_str}")

#     raw_score = score if em_check(answer, ground_truth["target"]) else format_score

#     return raw_score
#     # n_tool_calls = len(get_tool_call_contents(solution_str))
#     # return raw_score / (n_tool_calls + 1)


# def compute_score_subem(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
#     """子串匹配 (SubEM) 评分"""
#     is_legal, has_unclosed = check_legality(solution_str)
    
#     # 对不闭合的tool_call给予-1的严厉惩罚
#     if has_unclosed:
#         return -1
    
#     if not is_legal:
#         return 0

#     answer = extract_solution(solution_str)
#     do_print = random.randint(1, 64) == 1
#     if do_print:
#         print("--------------------------------")
#         print(f"Golden answers: {ground_truth['target']}")
#         print(f"Extracted answer: {answer}")
#         print(f"Solution string: {solution_str}")

#     raw_score = score if subem_check(answer, ground_truth["target"]) else format_score

#     # tool_call 数量惩罚
#     n_tool_calls = len(get_tool_call_contents(solution_str))
#     return raw_score / (n_tool_calls + 1)