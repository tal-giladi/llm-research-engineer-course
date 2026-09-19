"""Pretraining data loading: pack documents into one id stream, then cut
fixed-length ``(x, y)`` training windows out of it.

This is the data pipeline of Module 4 (`lessons/module-04/lesson-03.md`) and the
one the training loop in Module 7 calls each step. The whole corpus is encoded
once (by `llmre.tokenizer.bpe.BPETokenizer`) into a single 1-D tensor of token
ids, with documents separated by an end-of-text id. Each training example is a
``block_size``-long slice ``x`` together with the same slice shifted one
position to the left, ``y`` — so position ``t`` of ``x`` predicts ``y[t]``,
which is ``x[t + 1]``. That "predict the next token" shift is the entire
supervision signal for pretraining.
"""

from __future__ import annotations

import torch


def pack_documents(list_of_id_lists: list[list[int]], eot_id: int) -> torch.Tensor:
    """Concatenate encoded documents into one id stream, separated by ``eot_id``.

    Each document's ids are followed by a single ``eot_id`` marker, so the model
    sees an explicit boundary between documents and does not learn to "run on"
    from the end of one document into the start of an unrelated one.

    Args:
        list_of_id_lists: one list of ``int`` token ids per document.
        eot_id: the id of the ``<|endoftext|>`` separator token appended after
            every document.

    Returns:
        A 1-D ``torch.long`` tensor on the CPU of length
        ``sum(len(doc) + 1 for doc in list_of_id_lists)``.
    """
    stream: list[int] = []
    for doc in list_of_id_lists:
        stream.extend(doc)
        stream.append(eot_id)
    return torch.tensor(stream, dtype=torch.long)


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of ``(x, y)`` next-token training windows.

    ``batch_size`` start positions are drawn uniformly at random from the id
    stream. For each start ``i`` the input window is ``data[i : i + block_size]``
    and the target window is that same slice shifted one to the right,
    ``data[i + 1 : i + block_size + 1]``. Therefore ``y[b, t] == x[b, t + 1]``
    for every position ``t`` before the last: the target at each position is
    simply the next input token.

    Args:
        data: a 1-D ``torch.long`` tensor of token ids (e.g. from
            ``pack_documents``). Must have length ``> block_size``.
        block_size: context length ``T`` — the number of tokens per window.
        batch_size: number of windows ``B`` to stack into the batch.
        device: device string for the returned tensors (e.g. ``"cpu"`` or
            ``"cuda"``). Sampling happens on CPU; the batch is moved at the end.

    Returns:
        ``(x, y)``, each a ``torch.long`` tensor of shape ``(batch_size,
        block_size)`` on ``device``.
    """
    if data.dim() != 1:
        raise ValueError(f"data must be 1-D, got shape {tuple(data.shape)}")
    if data.numel() <= block_size:
        raise ValueError(
            f"data length {data.numel()} must exceed block_size {block_size}"
        )

    # Highest valid start index: we need block_size + 1 tokens (window + its shift).
    high = data.numel() - block_size
    ix = torch.randint(low=0, high=high, size=(batch_size,))

    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)
