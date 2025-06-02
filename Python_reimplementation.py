# main.py

import openai
import os
from collections import defaultdict
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Split body predicates properly (smart split — commas outside parentheses)
def split_predicates(body_text):
    return [x.strip() for x in re.split(r',\s*(?![^()]*\))', body_text)]

# 1. LLM Translation
def llm_translate(nl_text):
    system_prompt = (
        "Translate the following facts written in natural language into Prolog-style logic."
        " Also, define rules if needed."
        " Ensure that each predicate and rule is properly closed with parentheses and a period at the end."
        " Do not use Markdown formatting or code blocks."
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": nl_text}
        ],
        temperature=0
    )
    return response['choices'][0]['message']['content']

# 2. Check Logic Validity
def check_logic_validity(logic_text):
    errors = []
    for line_num, line in enumerate(logic_text.strip().split('\n'), start=1):
        stripped = line.strip()
        if not stripped.endswith('.'):
            errors.append(f"Line {line_num}: Missing period at end.")
        if '(' not in stripped or ')' not in stripped:
            errors.append(f"Line {line_num}: Missing parentheses.")
    return errors

# 3. Self-Refinement
def llm_self_refine(broken_logic, error_messages):
    error_prompt = "\n".join(error_messages)
    system_prompt = f"The following logic has errors:\n{error_prompt}\nPlease fix the logic without using Markdown code blocks."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": broken_logic}
        ],
        temperature=0
    )
    return response['choices'][0]['message']['content']

# 4. Parse Facts and Rules
def parse_logic(logic_text):
    facts = []
    rules = []

    logic_text = logic_text.replace('```prolog', '').replace('```', '')

    lines = logic_text.strip().split('\n')
    current_rule = ""
    for line in lines:
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        if ':-' in line or current_rule:
            current_rule += " " + line.strip()
            if current_rule.strip().endswith('.'):
                rules.append(current_rule.strip().strip('.'))
                current_rule = ""
        else:
            if '(' in line and ')' in line:
                predicate, args = line.strip('.').split('(')
                args = args.strip(')').split(',')
                facts.append((predicate.strip(), [arg.strip() for arg in args]))
            else:
                print(f"Skipping invalid line: {line}")

    return facts, rules

# 5. Dynamic Reasoning Engine with Consistent Variable Binding
from itertools import product

def apply_rules(facts, rules):
    fact_dict = defaultdict(list)
    for predicate, args in facts:
        fact_dict[predicate].append(args)

    derived_facts = set()

    for rule in rules:
        head, body = rule.split(':-')
        head_predicate, head_args = head.strip().split('(')
        head_args = [h.strip() for h in head_args.strip(')').split(',')]

        body = body.strip()
        if body.startswith('(') and body.endswith(')'):
            body = body[1:-1]

        body_predicates = split_predicates(body)

        body_preds = []
        for b in body_predicates:
            if '(' in b and ')' in b:
                pred_name, pred_args = b.split('(')
                pred_args = [arg.strip() for arg in pred_args.strip(')').split(',')]
                body_preds.append((pred_name.strip(), pred_args))
            else:
                print(f"Skipping invalid body predicate: {b}")

        if len(body_preds) == 0:
            print(f"Skipping rule (no valid body predicates): {rule}")
            continue

        # Cross-product of all possible matching facts
        fact_sets = []
        for pred_name, pred_vars in body_preds:
            fact_sets.append(fact_dict.get(pred_name, []))

        for fact_combo in product(*fact_sets):
            var_bindings = {}
            match = True
            for (pred_vars, fact_args) in zip([bp[1] for bp in body_preds], fact_combo):
                for var, val in zip(pred_vars, fact_args):
                    if var in var_bindings:
                        if var_bindings[var] != val:
                            match = False
                            break
                    else:
                        var_bindings[var] = val
                if not match:
                    break
            if match:
                result = tuple(var_bindings.get(var.strip(), '?') for var in head_args)
                derived_facts.add((head_predicate, result))

    return derived_facts

# 6. Main Pipeline
def main():
    natural_language_kb = """
John is the parent of Mary.
Tom is the parent of Alice.
Mary is the sibling of Tom.
Alice is the ancestor of Emma.
Define uncle as someone who is a sibling of a parent.
"""

    print("Translating natural language KB into logic...")
    logic_output = llm_translate(natural_language_kb)
    print("\nGenerated Logic:\n", logic_output)

    errors = check_logic_validity(logic_output)
    if errors:
        print("\nErrors found in logic:")
        for err in errors:
            print(err)
        logic_output = llm_self_refine(logic_output, errors)
        print("\nRefined Logic:\n", logic_output)
    else:
        print("\nLogic is valid. No refinement needed.")

    facts, rules = parse_logic(logic_output)
    print("\nParsed Facts:", facts)
    print("Parsed Rules:", rules)

    derived_facts = apply_rules(facts, rules)
    print("\nDerived Facts:")
    for pred, args in derived_facts:
        print(f"{pred}({', '.join(args)})")

if __name__ == "__main__":
    main()
