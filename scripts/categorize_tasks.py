#!/usr/bin/env python3
"""
Categorize all 164 HumanEval tasks into task types.

This script loads the HumanEval dataset and categorizes each task using:
1. Existing manual categorization (50 tasks)
2. Keyword-based heuristics for remaining tasks
3. Outputs to data/humaneval_categories_full.json
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets import load_dataset

# Existing manual categorization from pilot (50 tasks)
MANUAL_CATEGORIZATION = {
    "HumanEval/0": "list",      # has_close_elements (checks list elements)
    "HumanEval/1": "string",    # separate_paren_groups
    "HumanEval/2": "math",      # truncate_number
    "HumanEval/3": "list",      # below_zero
    "HumanEval/4": "math",      # mean_absolute_deviation
    "HumanEval/5": "list",      # intersperse
    "HumanEval/6": "string",    # parse_nested_parens
    "HumanEval/7": "list",      # filter_by_substring
    "HumanEval/8": "math",      # sum_product
    "HumanEval/9": "list",      # rolling_max
    "HumanEval/10": "string",   # is_palindrome (string check)
    "HumanEval/11": "string",   # string_xor
    "HumanEval/12": "string",   # longest
    "HumanEval/13": "math",     # greatest_common_divisor
    "HumanEval/14": "list",     # all_prefixes
    "HumanEval/15": "string",   # string_sequence
    "HumanEval/16": "math",     # count_distinct_characters
    "HumanEval/17": "string",   # parse_music
    "HumanEval/18": "math",     # how_many_times
    "HumanEval/19": "string",   # sort_numbers (words to nums)
    "HumanEval/20": "list",     # find_closest_elements
    "HumanEval/21": "list",     # rescale_to_unit
    "HumanEval/22": "list",     # filter_integers
    "HumanEval/23": "math",     # strlen (counting)
    "HumanEval/24": "math",     # largest_divisor
    "HumanEval/25": "math",     # factorize
    "HumanEval/26": "list",     # remove_duplicates
    "HumanEval/27": "string",   # flip_case
    "HumanEval/28": "string",   # concatenate
    "HumanEval/29": "list",     # filter_by_prefix
    "HumanEval/30": "list",     # get_positive
    "HumanEval/31": "math",     # is_prime
    "HumanEval/32": "math",     # poly (polynomial)
    "HumanEval/33": "list",     # sort_third
    "HumanEval/34": "list",     # unique
    "HumanEval/35": "math",     # max_element
    "HumanEval/36": "math",     # fizz_buzz
    "HumanEval/37": "list",     # sort_even
    "HumanEval/38": "string",   # decode_cyclic
    "HumanEval/39": "math",     # prime_fib
    "HumanEval/40": "list",     # triples_sum_to_zero
    "HumanEval/41": "math",     # car_race_collision
    "HumanEval/42": "list",     # incr_list
    "HumanEval/43": "list",     # pairs_sum_to_zero
    "HumanEval/44": "math",     # change_base
    "HumanEval/45": "math",     # triangle_area
    "HumanEval/46": "math",     # fib4
    "HumanEval/47": "math",     # median
    "HumanEval/48": "logic",    # is_palindrome (logic variant)
    "HumanEval/49": "math",     # modp
}

# Keyword patterns for classification
CATEGORY_KEYWORDS = {
    "string": [
        "string", "char", "character", "letter", "word", "text", "parse",
        "vowel", "consonant", "uppercase", "lowercase", "substring", "encode",
        "decode", "palindrome", "anagram", "bracket", "paren", "concat"
    ],
    "math": [
        "sum", "product", "number", "prime", "digit", "factor", "divide",
        "multiply", "add", "subtract", "calculate", "fibonacci", "fib",
        "gcd", "lcm", "modulo", "power", "square", "root", "mean", "median",
        "average", "count", "triangle", "area", "perimeter", "geometric",
        "arithmetic", "even", "odd", "integer", "decimal", "float", "round"
    ],
    "list": [
        "list", "array", "sort", "filter", "element", "remove", "insert",
        "append", "pop", "index", "slice", "reverse", "rotate", "max",
        "min", "unique", "duplicate", "pairs", "triple", "tuple", "vector",
        "sequence", "subsequence", "prefix", "suffix", "merge", "split"
    ],
    "logic": [
        "check", "valid", "verify", "condition", "boolean", "true", "false",
        "match", "bracket", "balanced", "nested", "recursive", "algorithm"
    ],
    "search": [
        "find", "search", "closest", "nearest", "binary", "linear",
        "lookup", "position", "locate", "exist", "contain"
    ]
}


def classify_task(prompt: str, entry_point: str) -> str:
    """
    Classify a task based on its prompt and entry point.
    
    Args:
        prompt: Task prompt/description
        entry_point: Function name
        
    Returns:
        Category string: string, math, list, logic, or search
    """
    text = (prompt + " " + entry_point).lower()
    
    # Score each category based on keyword matches
    scores = {cat: 0 for cat in CATEGORY_KEYWORDS}
    
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                scores[category] += 1
    
    # Get category with highest score
    if max(scores.values()) > 0:
        return max(scores, key=scores.get)
    
    # Default to logic for ambiguous cases
    return "logic"


def main():
    print("Loading HumanEval dataset...")
    dataset = load_dataset("openai_humaneval", split="test")
    
    categorization = {}
    stats = {"string": 0, "math": 0, "list": 0, "logic": 0, "search": 0}
    manual_count = 0
    auto_count = 0
    
    print(f"Processing {len(dataset)} tasks...")
    
    for item in dataset:
        task_id = item["task_id"]
        
        # Use manual categorization if available
        if task_id in MANUAL_CATEGORIZATION:
            category = MANUAL_CATEGORIZATION[task_id]
            manual_count += 1
        else:
            # Auto-categorize
            category = classify_task(item["prompt"], item["entry_point"])
            auto_count += 1
        
        categorization[task_id] = category
        stats[category] += 1
    
    # Sort by task ID number
    sorted_cats = dict(sorted(
        categorization.items(),
        key=lambda x: int(x[0].split("/")[1])
    ))
    
    # Save to JSON
    output_path = Path(__file__).parent.parent / "data" / "humaneval_categories_full.json"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(sorted_cats, f, indent=2)
    
    print(f"\n✓ Categorized {len(categorization)} tasks:")
    print(f"  - Manual: {manual_count}")
    print(f"  - Auto:   {auto_count}")
    print(f"\nCategory distribution:")
    for cat, count in sorted(stats.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(categorization)
        print(f"  - {cat}: {count} ({pct:.1f}%)")
    
    print(f"\n✓ Saved to: {output_path}")


if __name__ == "__main__":
    main()
