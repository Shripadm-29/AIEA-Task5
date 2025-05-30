import openai
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

import os
openai.api_key = os.getenv("OPENAI_API_KEY")

# 1. LLM Translation: Natural language KB -> Prolog-style logic
def llm_translate(nl_text):
    system_prompt = "Translate the following facts into Prolog-style logic and add a grandparent rule."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": nl_text}
        ],
        temperature=0
    )
    return response['choices'][0]['message']['content']

# 2. Simple Syntax Validation
def is_logic_valid(logic_text):
    for line in logic_text.strip().split('\n'):
        if not line.strip().endswith('.'):
            return False, "Missing period at end of statement."
    return True, ""

# 3. LLM Self-Refinement (fix logic errors)
def llm_self_refine(broken_logic, error_message):
    system_prompt = f"The following logic has an error: {error_message}. Please fix it."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": broken_logic}
        ],
        temperature=0
    )
    return response['choices'][0]['message']['content']

# 4. Parse Logic Facts
def parse_logic(logic_text):
    facts = []
    rule = None
    for line in logic_text.strip().split('\n'):
        line = line.strip().strip('.')
        if line.startswith("parent"):
            fact = line[len("parent("):-1]
            parent, child = [item.strip() for item in fact.split(',')]
            facts.append((parent, child))
        elif line.startswith("grandparent"):
            rule = "grandparent(X, Y) :- parent(X, Z), parent(Z, Y)."
    return facts, rule

# 5. Symbolic Solver
def find_grandparents(facts):
    children_of = defaultdict(list)
    for p, c in facts:
        children_of[p].append(c)

    grandparents = set()
    for p1 in children_of:
        for child in children_of[p1]:
            for grandchild in children_of.get(child, []):
                grandparents.add((p1, grandchild))
    return grandparents

# 6. Main Pipeline
def main():
    natural_language_kb = """
John is the parent of Mary.
Mary is the parent of Susan.
John is the parent of Mike.
Mike is the parent of Kevin.
Lisa is the parent of John.
James is the parent of Lisa.
Mary is the parent of Tom.
Mike is the parent of Alice.
Alice is the parent of Emma.
Tom is the parent of Rachel.
"""

    # Step 1: LLM Translation
    print("Translating natural language to logic...")
    logic_output = llm_translate(natural_language_kb)
    print("\nGenerated Logic:\n", logic_output)

    # Step 2: Validate Logic
    valid, error_message = is_logic_valid(logic_output)

    # Step 3: If Not Valid, Refine Logic
    if not valid:
        print("\nError detected:", error_message)
        logic_output = llm_self_refine(logic_output, error_message)
        print("\nRefined Logic:\n", logic_output)
    else:
        print("\nLogic is valid. No refinement needed.")

    # Step 4: Parse Logic
    facts, rule = parse_logic(logic_output)
    print("\nParsed Facts:", facts)
    print("Parsed Rule:", rule)

    # Step 5: Reasoning
    grandparents = find_grandparents(facts)
    print("\nGrandparent Relationships Found:")
    for gp, gc in grandparents:
        print(f"{gp} is the grandparent of {gc}")

if __name__ == "__main__":
    main()
