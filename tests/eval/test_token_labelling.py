from pathlib import Path

import pytest

import delphi.eval.token_labelling as tl
from delphi.constants import STATIC_ASSETS_DIR


def compare_list_of_dicts(list1, list2):
    if len(list1) != len(list2):
        return False

    # Compare each dictionary in the sorted lists
    for dict1, dict2 in zip(list1, list2):
        if dict1 != dict2:
            return False

    return True


def test_token_labeling():
    # Generate the original list of dictionaries
    original_list_of_dicts = tl.label_tokens_from_tokenizer()

    # Export to CSV (I'm assuming you have a function for this in your module)
    # export_path = Path("path/to/your/STATIC_ASSETS_DIR/token_labels.csv")
    tl.export_csv(original_list_of_dicts)

    returned_list_of_dicts = tl.import_csv_as_list_of_dicts(
        STATIC_ASSETS_DIR / "token_labels.csv"
    )

    assert compare_list_of_dicts(
        original_list_of_dicts, returned_list_of_dicts
    ), "The dictionaries are not the same after exporting and importing."
