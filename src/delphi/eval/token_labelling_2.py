from typing import Callable, Optional

TOKEN_LABELS: dict[str, Callable] = {
    # --- custom categories ---
    "Starts with space": (lambda token: token.text.startswith(" ")),  # bool
    "Capitalized": (
        lambda token: token.text[0].isupper()
    ),  # bool  #could be in the 2nd?
}
