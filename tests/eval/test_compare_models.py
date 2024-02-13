import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from delphi.eval.compare_models import NextTokenStats, compare_models
from delphi.eval.utils import load_validation_dataset, tokenize


def test_compare_models():
    with torch.set_grad_enabled(False):
        model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
        model_instruct = AutoModelForCausalLM.from_pretrained(
            "roneneldan/TinyStories-Instruct-1M"
        )
        ds_txt = load_validation_dataset("tinystories-v2-clean")["story"]
        tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-1M")
        sample_tok = tokenize(tokenizer, ds_txt[0])
        K = 3
        model_comparison = compare_models(model, model_instruct, sample_tok, top_k=K)
        # ignore the first element comparison
        assert model_comparison[0] is None
        assert isinstance(model_comparison[1], NextTokenStats)
        assert len(model_comparison) == sample_tok.shape[0]
        assert len(model_comparison[1].topk) == K
