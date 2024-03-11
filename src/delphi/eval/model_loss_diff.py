# TODO: write tests


def next_logprobs_to_loss(next_logprobs: list[list[float]]) -> list[list[float]]:
    """
    Given a list of log probabilities, calculate the loss for each token.

    args:
    - next_logprobs: a list of lists of log probabilities, e.g. [[-0.1, -0.2, ...], [-0.3, -0.4, ...], ...]

    returns: a list of lists of losses, e.g. [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]
    """
    next_loss = []
    for logprobs in next_logprobs:
        loss = []
        for logprob in logprobs:
            if logprob is None:
                loss.append(0.0)
            else:
                # loss is the negative log probability
                loss.append(-logprob)
        next_loss.append(loss)
    return next_loss


def next_logprobs_dict_to_loss_dict(
    next_logprobs: dict[str, list[list[float]]]
) -> dict[str, list[list[float]]]:
    """
    Given a dictionary of log probabilities mapping to models, calculate the loss for each token and create an appropriate dictionary of losses.

    args:
    - next_logprobs: a dictionary of lists of log probabilities, e.g. {"llama2-100k": [[-0.1, -0.2, ...], [-0.3, -0.4, ...], ...], ...}

    returns: a dictionary of lists of losses, e.g. {"llama2-100k": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...], ...}
    """
    next_loss = {}
    for key, logprobs in next_logprobs.items():
        next_loss[key] = next_logprobs_to_loss(logprobs)
    return next_loss
