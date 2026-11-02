"""Weakly supervised MIL training: slide labels only, one bag per step (CLAM defaults).

Bags are (features (N, D) tensor, slide label int). Patch labels and lesion ids never enter this module.
"""

import copy
import math
from collections.abc import Callable

import torch
from torch import nn


class EarlyStopping:
    """Stop when val loss has not improved for ``patience`` epochs, but never before ``stop_epoch``."""

    def __init__(self, patience: int = 20, stop_epoch: int = 50):
        self.patience, self.stop_epoch = patience, stop_epoch
        self.best = math.inf
        self.counter = 0

    def step(self, epoch: int, val_loss: float) -> tuple[bool, bool]:
        """Returns (improved, stop)."""
        if val_loss < self.best:
            self.best, self.counter = val_loss, 0
            return True, False
        self.counter += 1
        return False, self.counter >= self.patience and epoch > self.stop_epoch


def to_device(x, device):
    """A bag input (a tensor, or a tuple of tensors such as (features, grid)) on ``device``."""
    return tuple(t.to(device) for t in x) if isinstance(x, tuple) else x.to(device)


def evaluate(model: nn.Module, bags, device="cpu") -> tuple[float, list[float]]:
    """Mean cross-entropy and P(tumor) per bag."""
    model.eval()
    loss_fn, total, probs = nn.CrossEntropyLoss(), 0.0, []
    with torch.no_grad():
        for x, y in bags:
            logits, _ = model(to_device(x, device))
            total += loss_fn(logits[None], torch.tensor([y], device=device)).item()
            probs.append(torch.softmax(logits, dim=0)[1].item())
    return total / max(len(bags), 1), probs


def train_mil(build_model: Callable[[], nn.Module], train_bags, val_bags, tcfg: dict, seed: int,
              on_epoch: Callable[[nn.Module, int], None] | None = None,
              extra_loss: Callable[[nn.Module, int], torch.Tensor] | None = None, device="cpu"):
    """Train with Adam + early stopping on val loss; returns (best model, history).

    The model is built after seeding so its initial weights are reproducible too. ``on_epoch(model, epoch)``
    runs before each epoch (e.g. a temperature schedule). ``extra_loss(model, y)`` is added to each training
    step's loss (e.g. auxiliary heads); validation loss stays the slide cross-entropy. Bags stay where they
    are and move to ``device`` one at a time.
    """
    torch.manual_seed(seed)
    model = build_model().to(device)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    stopper = EarlyStopping(tcfg["patience"], tcfg["stop_epoch"])
    best_state, history = copy.deepcopy(model.state_dict()), []

    for epoch in range(tcfg["max_epochs"]):
        if on_epoch is not None:
            on_epoch(model, epoch)
        model.train()
        train_loss = 0.0
        for i in torch.randperm(len(train_bags), generator=g).tolist():
            x, y = train_bags[i]
            logits, _ = model(to_device(x, device))
            loss = loss_fn(logits[None], torch.tensor([y], device=device))
            if extra_loss is not None:
                loss = loss + extra_loss(model, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
        val_loss, _ = evaluate(model, val_bags, device)
        improved, stop = stopper.step(epoch, val_loss)
        if improved:
            best_state = copy.deepcopy(model.state_dict())
        history.append({"epoch": epoch, "train_loss": train_loss / len(train_bags), "val_loss": val_loss})
        if stop:
            break

    model.load_state_dict(best_state)
    return model, history
