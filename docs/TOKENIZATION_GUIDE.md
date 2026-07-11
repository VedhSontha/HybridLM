# Tokenization Preprocessing Guide

Subword tokenization parameters for RAG retrieval inputs:
- Tokenizer: BERT subword tokenizer.
- Padding: max length constrained to 512 tokens with truncation enabled.
- Vocabulary file loaded locally to minimize remote network lookups.
