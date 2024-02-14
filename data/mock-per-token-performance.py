from math import log2
from random import random, seed

seed(0)

performance_data = {
    "llama2-100k": {
        "nouns": [log2(random()) for _ in range(200)],
        "verbs": [log2(random()) for _ in range(100)],
        "adjectives": [log2(random()) for _ in range(250)],
        "adverbs": [log2(random()) for _ in range(40)],
        "pronouns": [log2(random()) for _ in range(55)],
        "prepositions": [log2(random()) for _ in range(234)],
    },
    "llama2-1m": {
        "nouns": [log2(random()) for _ in range(200)],
        "verbs": [log2(random()) for _ in range(100)],
        "adjectives": [log2(random()) for _ in range(250)],
        "adverbs": [log2(random()) for _ in range(40)],
        "pronouns": [log2(random()) for _ in range(55)],
        "prepositions": [log2(random()) for _ in range(234)],
    },
    "llama2-10m": {
        "nouns": [log2(random()) for _ in range(200)],
        "verbs": [log2(random()) for _ in range(100)],
        "adjectives": [log2(random()) for _ in range(250)],
        "adverbs": [log2(random()) for _ in range(40)],
        "pronouns": [log2(random()) for _ in range(55)],
        "prepositions": [log2(random()) for _ in range(234)],
    },
    "llama2-100m": {
        "nouns": [log2(random()) for _ in range(200)],
        "verbs": [log2(random()) for _ in range(100)],
        "adjectives": [log2(random()) for _ in range(250)],
        "adverbs": [log2(random()) for _ in range(40)],
        "pronouns": [log2(random()) for _ in range(55)],
        "prepositions": [log2(random()) for _ in range(234)],
    },
    "llama2-1b": {
        "nouns": [log2(random()) for _ in range(200)],
        "verbs": [log2(random()) for _ in range(100)],
        "adjectives": [log2(random()) for _ in range(250)],
        "adverbs": [log2(random()) for _ in range(40)],
        "pronouns": [log2(random()) for _ in range(55)],
        "prepositions": [log2(random()) for _ in range(234)],
    },
}
