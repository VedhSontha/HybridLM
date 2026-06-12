# modules/preprocess.py
"""
HybridLM NLP Preprocessing module
--------------------------------
Provides a deterministic, entity-aware preprocessing pipeline:
- Unicode normalization
- Contraction expansion
- Placeholder masking (<URL>, <EMAIL>, <NUM>, <CODE>)
- Entity-Aware Semantic Restoration (EASR) for corrupted entities & dates
- Rule-based tokenization + configurable stopword filtering
- spaCy analysis: tokens, lemmas, POS, NER, sentences
- Unified JSON output via `process(text)`

Usage:
    from modules.preprocess import TextPreprocessor
    tp = TextPreprocessor()
    out = tp.process("messy input text...")
"""

from __future__ import annotations
import re
import unicodedata
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Try to import spaCy lazily so import failure is informative.
try:
    import spacy
    from spacy.language import Language
except Exception as e:
    spacy = None  # We'll raise helpful errors in runtime if spaCy not available.


# -------------------------
# Configuration / resources
# -------------------------
CONTRACTIONS = {
    "don't": "do not", "can't": "cannot", "i'm": "i am", "it's": "it is",
    "he's": "he is", "she's": "she is", "they're": "they are", "we're": "we are",
    "let's": "let us", "didn't": "did not", "won't": "will not", "couldn't": "could not",
}

URL_RE = re.compile(r'https?://\S+|www\.\S+', flags=re.IGNORECASE)
EMAIL_RE = re.compile(r'\S+@\S+')
NUM_RE = re.compile(r'\b\d+(?:[.,]\d+)?\b')
PUNCT_RE = re.compile(r'[^\w\s<>]')
WHITESPACE_RE = re.compile(r'\s+')

# Token regex: keep placeholders, words with apostrophes, and single punctuation tokens
TOK_RE = re.compile(r"(<URL>|<EMAIL>|<NUM>|<CODE>|[A-Za-z0-9']+|[^\s\w])")

# Abbrev list for sentence splitting
ABBREVIATIONS = {"mr.", "mrs.", "dr.", "ms.", "u.s.", "e.g.", "i.e.", "etc."}

# -------------------------
# Optional sequence encoder
# -------------------------
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


class SequenceEncoder(nn.Module if TORCH_AVAILABLE else object):
    """
    Optional small BiLSTM-based sequence encoder.
    Returns a summary vector (torch.Tensor) for given token embeddings.
    Only usable if PyTorch is installed.
    """
    def __init__(self, input_dim: int = 300, hidden_dim: int = 512,
                 num_layers: int = 1, bidirectional: bool = True):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available. Install torch to use SequenceEncoder.")
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=bidirectional, batch_first=True)
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (batch, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)
        if self.bidirectional:
            # last two layers contain forward/backward final states
            h = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            h = h_n[-1, :, :]
        return h  # shape (batch, hidden_dim * (2 if bidirectional else 1))


# -------------------------
# TextPreprocessor class
# -------------------------
class TextPreprocessor:
    """
    Main preprocessing pipeline for HybridLM.

    Args:
        spacy_model: spaCy model name (default 'en_core_web_sm')
        lowercase: whether to lowercase during cleaning (kept off during EASR restore if needed)
        remove_urls: whether to mask URLs and emails as placeholders
        num_token: placeholder token for numbers (default '<NUM>')
        stopword_mode: 'none'|'light'|'all' -- controls stopword filtering for token list
        enable_lstm_summary: if True and torch available, a SequenceEncoder is created (not automatically used)
    """

    def __init__(self,
                 spacy_model: str = "en_core_web_sm",
                 lowercase: bool = True,
                 remove_urls: bool = True,
                 num_token: str = "<NUM>",
                 stopword_mode: str = "light",
                 enable_lstm_summary: bool = False):
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.num_token = num_token
        self.stopword_mode = stopword_mode
        self.spacy_model = spacy_model
        self._nlp: Optional[Language] = None
        self.enable_lstm_summary = enable_lstm_summary

        if spacy is None:
            raise ImportError("spaCy is required by modules.preprocess. Install with `pip install spacy` "
                              "and download a model `python -m spacy download en_core_web_sm`.")

        # Lazy load spaCy model
        try:
            self._nlp = spacy.load(self.spacy_model, disable=["parser"])  # parser can be re-enabled if wanted
            # If we want sentence segmentation we'll enable parser, but we can also use sentencizer.
            if not self._nlp.has_pipe("sentencizer"):
                self._nlp.add_pipe("sentencizer")
        except Exception as e:
            logger.error("Failed to load spaCy model '%s': %s", self.spacy_model, e)
            raise

        if enable_lstm_summary:
            if not TORCH_AVAILABLE:
                raise RuntimeError("enable_lstm_summary requires PyTorch (torch).")
            # user can later create an encoder with desired dims externally
            self.lstm_encoder: Optional[SequenceEncoder] = SequenceEncoder(input_dim=300, hidden_dim=512)
        else:
            self.lstm_encoder = None

    # -------------------------
    # Basic cleaning helpers
    # -------------------------
    @staticmethod
    def normalize_unicode(text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        # remove zero-width characters
        text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
        return text

    @staticmethod
    def expand_contractions(text: str) -> str:
        for k, v in CONTRACTIONS.items():
            text = re.sub(r"\b" + re.escape(k) + r"\b", v, text, flags=re.IGNORECASE)
        return text

    # -------------------------
    # Entity-Aware Semantic Restoration (EASR)
    # -------------------------
    @staticmethod
    def restore_entities(text: str) -> str:
        """
        HybridLM EASR v4.1 — Final hybrid version.
        Fixes:
        • 'v e d h L L M' → HybridLM (spacing too loose)
        • 'gppT-5' / 'GPT5' → GPT-5 (dashless variants)
        • 's&p' → S&P (better grouping)
        """

        # --- MODEL & ORG RESTORATION ---
        text = re.sub(r"v[\s\.\-]*e[\s\.\-]*d[\s\.\-]*h[\s\.\-]*l[\s\.\-]*l[\s\.\-]*m+", "HybridLM", text, flags=re.IGNORECASE)
        text = re.sub(r"0pen[\.\-]*ai", "OpenAI", text, flags=re.IGNORECASE)
        text = re.sub(r"g+p+p*t[\s\-\_]*([0-9]+)", r"GPT-\1", text, flags=re.IGNORECASE)  # ← FIXED
        text = re.sub(r"medicalg+p*t", "MedicalGPT", text, flags=re.IGNORECASE)
        text = re.sub(r"ll[\_\-\s]*am[a@]", "LLaMA", text, flags=re.IGNORECASE)
        text = re.sub(r"fin[\-\s]*ai", "FinAI", text, flags=re.IGNORECASE)
        text = re.sub(r"dee+p*m[i1]nd", "DeepMind", text, flags=re.IGNORECASE)
        text = re.sub(r"google", "Google", text, flags=re.IGNORECASE)
        text = re.sub(r"\bs[\s\&]*p\b", "S&P", text, flags=re.IGNORECASE)  # ← FIXED

        # --- TEMPORAL NORMALIZATION ---
        text = re.sub(r"(?i)\bthe\s+year\s+after\s+2024\b", "2025", text)
        text = re.sub(r"(?i)\bthe\s+year\s+after\s+2023\b", "2024", text)
        text = re.sub(r"(?i)\b20\s*2\s*five\b", "2025", text)
        text = re.sub(r"(?i)\b20\s*2\s*four\b", "2024", text)
        text = re.sub(r"(?i)\b20\s*2\s*three\b", "2023", text)
        text = re.sub(r"(?i)\b'25\b", "2025", text)
        text = re.sub(r"(?i)\b'24\b", "2024", text)
        text = re.sub(r"(?i)\b'23\b", "2023", text)
        text = re.sub(r"(?i)around\s+the\s+mid[\-\s]*20s", "2025", text)
        text = re.sub(r"\b(20{0,1}25)\b", "2025", text)
        text = re.sub(r"\b(20{0,1}24)\b", "2024", text)
        text = re.sub(r"\b(20{0,1}23)\b", "2023", text)

        # --- HASHTAG + PUNCTUATION CLEANUP ---
        text = re.sub(r"#\s*ai", " AI", text, flags=re.IGNORECASE)
        text = re.sub(r"(\d)(AI)", r"\1 AI", text)
        text = re.sub(r"[!?]{2,}|…", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        # --- SMART CAPITALIZATION ---
        important_terms = [
            "HybridLM", "OpenAI", "MedicalGPT", "DeepMind", "Google",
            "GPT", "FinAI", "LLaMA", "S&P", "AI",
            "HTTP", "HTTPS", "URL", "API", "CPU", "GPU", "RAM", "USB"
        ]
        for term in important_terms:
            text = re.sub(term.lower(), term, text, flags=re.IGNORECASE)

        def smart_case(match):
            word = match.group(0)
            if word in important_terms or word.upper() in {"HTTP", "HTTPS", "URL", "API", "CPU", "GPU", "RAM", "USB"}:
                return word.upper() if word.isupper() else word
            return word.capitalize()

        text = re.sub(r'\b[A-Z]{2,}\b', smart_case, text)

        return text



    # -------------------------
    # Primary cleaning pipeline
    # -------------------------
    def clean_text(self, text: str,
                lowercase: Optional[bool] = None,
                remove_urls: Optional[bool] = None,
                num_token: Optional[str] = None) -> str:
        """
        Full cleaning pipeline that integrates normalization, optional masking, contraction expansion,
        and EASR restoration.
        
        CRITICAL ORDER:
        1. Unicode normalization (safe)
        2. Entity restoration (MUST be early, needs original patterns)
        3. URL/email masking
        4. Contraction expansion
        5. Lowercase (destroys case-sensitive patterns)
        6. Number masking (destroys temporal patterns)
        7. Punctuation cleanup
        """
        if lowercase is None:
            lowercase = self.lowercase
        if remove_urls is None:
            remove_urls = self.remove_urls
        if num_token is None:
            num_token = self.num_token

        # Step 1: Unicode normalization (safe, preserves all text)
        text = self.normalize_unicode(text)
        
        # Step 2: Entity restoration FIRST (needs original patterns)
        # This MUST happen before lowercase and number masking!
        text = self.expand_contractions(text)
        
        # Step 3: Optional URL/email masking (happens after entity restoration)
        if remove_urls:
            text = URL_RE.sub("<URL>", text)
            text = EMAIL_RE.sub("<EMAIL>", text)

        # Step 4: Contraction expansion
        text = re.sub(r"(?<!\d)\b(\d{1,2})\b(?!\d)", num_token, text)

        text = self.restore_entities(text)

        
        

        protected_terms = [
            "HybridLM", "OpenAI", "MedicalGPT", "DeepMind", "Google",
            "GPT", "LLaMA", "FinAI", "S&P", "AI", "HTTP", "CPU", "GPU", "URL"
        ]
        
        if lowercase:
            tokens = text.split()
            text = " ".join([
                t if any(term in t for term in protected_terms)
                else t.lower()
                for t in tokens
            ])
        

        # Step 7: Remove odd punctuation but preserve placeholders like <URL> <EMAIL> <NUM>
        text = PUNCT_RE.sub(" ", text)
        text = WHITESPACE_RE.sub(" ", text).strip()

        return text

    # -------------------------
    # Tokenization & stopword filtering
    # -------------------------
    def custom_tokenize(self, text: str) -> List[str]:
        tokens = TOK_RE.findall(text)
        return [t for t in tokens if t.strip()]

    def filter_stopwords(self, tokens: List[str], mode: Optional[str] = None) -> List[str]:
        """
        mode:
          - 'none': keep all tokens
          - 'light': remove common stopwords except pronouns 'i', 'you', 'we'
          - 'all': remove all stopwords
        """
        if mode is None:
            mode = self.stopword_mode
        if mode == "none":
            return tokens
        # spaCy stopwords
        stopwords = set()
        try:
            stopwords = set(self._nlp.Defaults.stop_words)
        except Exception:
            stopwords = set()
        filtered = []
        for t in tokens:
            tl = t.lower()
            if mode == "all" and tl in stopwords:
                continue
            if mode == "light" and tl in stopwords and tl not in {"i", "you", "we"}:
                continue
            filtered.append(t)
        return filtered

    # -------------------------
    # spaCy analysis + packaging
    # -------------------------
    def analyze(self, text: str) -> Tuple[List[str], List[str], List[str], List[Dict[str, str]], List[str]]:
        """
        Run spaCy pipeline and return tokens, lemmas, pos_tags, entities (list of dicts), sentences.
        """
        doc = self._nlp(text)
        tokens = [t.text for t in doc]
        lemmas = [t.lemma_ for t in doc]
        pos_tags = [t.pos_ for t in doc]
        ents = [{"text": e.text, "label": e.label_} for e in doc.ents]
        sentences = [sent.text for sent in doc.sents]
        return tokens, lemmas, pos_tags, ents, sentences

    # -------------------------
    # Mapping helper to BERT tokenizer (optional)
    # -------------------------
    def tokens_to_bert(self, tokens: List[str], tokenizer, max_len: int = 512) -> Dict[str, Any]:
        """
        Map custom tokens to a HuggingFace tokenizer.
        Returns input_ids, attention_mask, and a mapping list:
          token_to_subtoken_map: List[List[int]] mapping each token index to the subtoken indices inside input_ids.
        Requires that the tokenizer is a HuggingFace tokenizer object.
        """
        # Build a string that preserves token boundaries by joining with space.
        text = " ".join(tokens)
        enc = tokenizer(text, truncation=True, padding="max_length", max_length=max_len, return_offsets_mapping=True)
        # Use offsets to map token -> subtoken indices. This mapping is heuristic and may need refinement.
        offsets = enc["offset_mapping"]
        input_ids = enc["input_ids"]
        token_to_subtoken = []
        token_spans = []
        cursor = 0
        for t in tokens:
            start = text.find(t, cursor)
            if start == -1:
                # fallback: mark no mapping
                token_to_subtoken.append([])
                continue
            end = start + len(t)
            # collect subtoken indexes that fall within start..end
            sub_idxs = [i for i, (s, e) in enumerate(offsets) if s >= start and e <= end]
            token_to_subtoken.append(sub_idxs)
            cursor = end
        return {"input_ids": input_ids, "attention_mask": enc["attention_mask"], "token_to_subtoken": token_to_subtoken}

    # -------------------------
    # Top-level process() that returns JSON-ready dict
    # -------------------------
    def process(self, text: str) -> Dict[str, Any]:
        """
        Full pipeline: restore, clean, analyze, package into structured dict.

        Returns:
            {
              "original_text": str,
              "clean_text": str,
              "sentences": List[str],
              "tokens": List[str],
              "lemmas": List[str],
              "pos_tags": List[str],
              "entities": List[{"text":..., "label":...}],
              "token_count": int
            }
        """
        if not isinstance(text, str):
            text = str(text)

        cleaned = self.clean_text(text)
        tokens_raw = self.custom_tokenize(cleaned)
        tokens = self.filter_stopwords(tokens_raw, self.stopword_mode)
        # use spaCy to extract lemmas/pos/ents/sentences from cleaned text
        spa_tokens, lemmas, pos_tags, ents, sentences = self.analyze(cleaned)

        result = {
            "original_text": text,
            "clean_text": cleaned,
            "sentences": sentences,
            "tokens": tokens,
            "spa_tokens": spa_tokens,
            "lemmas": lemmas,
            "pos_tags": pos_tags,
            "entities": ents,
            "token_count": len(tokens)
        }

        # optional LSTM summary if enabled and something else set up by user
        if self.lstm_encoder is not None:
            # user must feed embeddings to lstm externally; placeholder here
            result["lstm_summary"] = None

        return result


# -------------------------
# Module usage example (if run directly)
# -------------------------
if __name__ == "__main__":
    # Quick demonstration (not executed on import)
    tp = TextPreprocessor()
    examples = [
        "v e d h L L M launched by 0pen.AI in the year after 2024!!!",
        "medicalgppt (GOOGLE DEEPMIND) became popular after the year after 2023#AI???",
        "GPT-5 announced by OpenAI on https://openai.com in '25."
    ]
    for s in examples:
        out = tp.process(s)
        import json
        print(json.dumps(out, indent=2))
