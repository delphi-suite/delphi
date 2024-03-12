corpus_dataset = "delphi-suite/tinystories-v2-clean"
tokenized_corpus_dataset = "delphi-suite/tinystories-v2-clean-tokenized-v0"

LLAMA2_MODELS = [
    "llama2-100k",
    "llama2-200k",
    "llama2-400k",
    "llama2-800k",
    "llama2-1.6m",
    "llama2-3.2m",
    "llama2-6.4m",
    "llama2-12.8m",
    "llama2-25.6m",
]

CATEGORY_MAP = {
    "nouns": "Is Noun",
    "verbs": "Is Verb",
    "adjectives": "Is Adjective",
    "adverbs": "Is Adverb",
    "pronouns": "Is Pronoun",
    "proper_nouns": "Is Proper Noun",
}
