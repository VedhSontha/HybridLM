# tests/nlp_sanity_check.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import re
from modules.preprocess import TextPreprocessor

tp = TextPreprocessor()

tests = [
    # Test format: (input, expected_entities, expected_in_clean_text, min_tokens)
    {
        "name": "Entity-heavy OCR corruption",
        "input": "MEDICALGPPT (GOOGLE DEEPMIND) BECAME POPULAR AFTER THE YEAR AFTER 2024#AI???",
        "expected_entities": ["MedicalGPT", "Google DeepMind"],
        "expected_in_text": ["MedicalGPT", "Google", "DeepMind", "2025", "AI"],
        "min_tokens": 5
    },
    {
        "name": "Spaced entity + year corruption",
        "input": "v e d h L L M launched by 0pen.AI in the year after 2024!!!",
        "expected_entities": ["HybridLM", "OpenAI"],
        "expected_in_text": ["HybridLM", "OpenAI", "2025"],
        "min_tokens": 4
    },
    {
        "name": "Model version + URL + shorthand year",
        "input": "gppT-5 announced by 0penai on https://openai.com in '25.",
        "expected_entities": ["OpenAI"],
        "expected_in_text": ["GPT-5", "OpenAI", "2025"],
        "min_tokens": 3
    },
    {
        "name": "Finance terms + emoji",
        "input": "fin ai predicts s&p 500 trends using transformer models since 20 2 five 🚀",
        "expected_entities": ["FinAI"],
        "expected_in_text": ["FinAI", "S&P", "2025"],
        "min_tokens": 6
    },
    {
        "name": "Neutral sentence (baseline)",
        "input": "This is a test sentence for the NLP pipeline.",
        "expected_entities": [],
        "expected_in_text": [],
        "min_tokens": 4
    },
    {
        "name": "Short entity query",
        "input": "Who built HybridLM?",
        "expected_entities": ["HybridLM"],
        "expected_in_text": ["HybridLM"],
        "min_tokens": 2
    },
    {
        "name": "Mid-20s temporal pattern",
        "input": "AI adoption accelerated around the mid-20s with GPT models.",
        "expected_entities": [],
        "expected_in_text": ["2025", "GPT"],
        "min_tokens": 5
    },
    {
        "name": "Tech acronyms preservation",
        "input": "THE GPU ACCELERATED THE CPU PROCESSING OF HTTP REQUESTS.",
        "expected_entities": [],
        "expected_in_text": ["GPU", "CPU", "HTTP"],
        "min_tokens": 5
    }
]

def normalize_for_comparison(text):
    """Normalize text for flexible matching."""
    # Remove extra spaces, convert to lowercase for comparison
    return re.sub(r'\s+', ' ', text.lower()).strip()

def check_token_presence(token, text, case_sensitive=False):
    """Check if token is present in text with flexible matching."""
    if not case_sensitive:
        token = token.lower()
        text = text.lower()
    # Allow for variations like "GPT-5" vs "GPT - 5"
    token_pattern = re.escape(token).replace(r'\-', r'[\s\-]*')
    return bool(re.search(r'\b' + token_pattern + r'\b', text))

failed = []
warnings = []

print("=" * 60)
print("🧪 NLP PIPELINE SANITY CHECK")
print("=" * 60)

for i, test in enumerate(tests, 1):
    print(f"\n{'─' * 60}")
    print(f"TEST {i}: {test['name']}")
    print(f"{'─' * 60}")
    print(f"Input: {test['input']}")
    
    # Run preprocessing
    out = tp.process(test['input'])
    
    # Extract results
    clean_text = out.get("clean_text", "")
    entities = [e["text"] for e in out.get("entities", [])]
    token_count = out.get("token_count", 0)
    
    print(f"\n📊 Results:")
    print(f"  Clean text: {clean_text}")
    print(f"  Entities: {entities}")
    print(f"  Token count: {token_count}")
    
    # Validation
    issues = []
    
    # Check 1: Token count
    if token_count < test['min_tokens']:
        issues.append(f"Token count too low (expected >={test['min_tokens']}, got {token_count})")
    
    # Check 2: Expected entities detected
    for expected_ent in test['expected_entities']:
        if expected_ent not in entities:
            # Check if it's at least in clean_text (warning, not failure)
            if check_token_presence(expected_ent, clean_text):
                warnings.append((i, test['name'], f"Entity '{expected_ent}' not extracted but present in text"))
            else:
                issues.append(f"Expected entity '{expected_ent}' not found anywhere")
    
    # Check 3: Expected tokens in clean text
    for expected_token in test['expected_in_text']:
        if not check_token_presence(expected_token, clean_text):
            issues.append(f"Expected token '{expected_token}' not found in clean_text")
    
    # Report results
    if issues:
        print(f"\n❌ FAILED:")
        for issue in issues:
            print(f"   • {issue}")
        failed.append((i, test['name'], issues))
    else:
        print(f"\n✅ PASSED")
    
    # Show full output in verbose mode
    print(f"\n📋 Full output:")
    print(json.dumps(out, indent=2, ensure_ascii=False))

# Summary
print("\n" + "=" * 60)
print("📈 SUMMARY")
print("=" * 60)

if warnings:
    print(f"\n⚠️  {len(warnings)} Warning(s):")
    for test_num, test_name, warning in warnings:
        print(f"   Test {test_num} ({test_name}): {warning}")

if failed:
    print(f"\n❌ {len(failed)} Test(s) Failed:")
    for test_num, test_name, issues in failed:
        print(f"\n   Test {test_num}: {test_name}")
        for issue in issues:
            print(f"      • {issue}")
    print("\n🔴 Some tests failed. Please review the pipeline.")
    raise SystemExit(2)
else:
    print(f"\n✅ All {len(tests)} tests passed!")
    if warnings:
        print("   (with warnings - check entity extraction)")
    print("\n🚀 NLP pipeline is ready for RAG integration!")