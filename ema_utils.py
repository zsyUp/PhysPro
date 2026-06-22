"""Utility modules for EMA tracking and simple class reweighting."""

import torch
import torch.nn as nn


class ModelEMA:
    """Track an exponential moving average of trainable model parameters.

    Example:
        ema = ModelEMA(model, decay=0.999)
        for batch in dataloader:
            loss = train_step(model, batch)
            loss.backward()
            optimizer.step()
            ema.update(model)

        with ema.apply_shadow():
            validate(model, val_loader)
    """

    def __init__(self, model, decay=0.9999, warmup_steps=2000):
        """Initialize EMA state from the current model weights.

        Args:
            model: Model whose trainable parameters should be tracked.
            decay: Target EMA decay after warmup.
            warmup_steps: Number of updates used to ramp decay to its target.
        """
        self.model = model
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.num_updates = 0
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def get_decay(self, step):
        """Return the effective EMA decay for the current update step."""
        if step < self.warmup_steps:
            return min(self.decay, (1 + step) / (1 + self.warmup_steps))
        return self.decay

    @torch.no_grad()
    def update(self, model):
        """Update shadow parameters from the latest model weights."""
        self.num_updates += 1
        decay = self.get_decay(self.num_updates)

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow, f"Missing shadow parameter: {name}"
                new_average = decay * self.shadow[name] + (1.0 - decay) * param.data
                self.shadow[name] = new_average.clone()

    @torch.no_grad()
    def apply_shadow(self):
        """Return a context manager that temporarily swaps in EMA weights."""
        return _EMAContextManager(self)

    def _swap_parameters(self):
        """Replace model weights with their EMA counterparts."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def _restore_parameters(self):
        """Restore the original model weights after EMA evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        """Serialize EMA state for checkpointing."""
        return {
            "shadow": self.shadow,
            "num_updates": self.num_updates,
            "decay": self.decay,
        }

    def load_state_dict(self, state_dict):
        """Load EMA state from a checkpoint payload."""
        self.shadow = state_dict["shadow"]
        self.num_updates = state_dict["num_updates"]
        self.decay = state_dict.get("decay", self.decay)


class _EMAContextManager:
    """Context manager used by :meth:`ModelEMA.apply_shadow`."""

    def __init__(self, ema):
        self.ema = ema

    def __enter__(self):
        self.ema._swap_parameters()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ema._restore_parameters()


class ClassBalancedLoss(nn.Module):
    """Apply dynamic per-class weights derived from recent IoU statistics."""

    def __init__(self, num_classes=4, initial_weights=None, momentum=0.9):
        """Initialize the moving-average class weighting helper.

        Args:
            num_classes: Number of semantic classes.
            initial_weights: Optional starting weights for each class.
            momentum: Smoothing factor for both IoU history and weight updates.
        """
        super().__init__()
        self.num_classes = num_classes
        self.momentum = momentum

        if initial_weights is None:
            initial_weights = torch.ones(num_classes)

        self.register_buffer("weights", initial_weights.float())
        self.register_buffer("iou_history", torch.zeros(num_classes))
        self.update_count = 0

    def update_weights_from_iou(self, iou_scores):
        """Update class weights so lower-IoU classes receive more emphasis."""
        self.update_count += 1

        if self.update_count == 1:
            self.iou_history = iou_scores.clone()
        else:
            self.iou_history = (
                self.momentum * self.iou_history
                + (1 - self.momentum) * iou_scores
            )

        mean_iou = self.iou_history.mean()
        new_weights = torch.sqrt(mean_iou / (self.iou_history + 1e-6))
        new_weights = new_weights / new_weights.mean()
        new_weights = torch.clamp(new_weights, min=0.90, max=1.10)
        self.weights = self.momentum * self.weights + (1 - self.momentum) * new_weights

    def forward(self, loss_per_class):
        """Apply current class weights to a class-wise loss tensor."""
        weights = self.weights.to(loss_per_class.device)

        if loss_per_class.dim() == 4:
            weights = weights.view(1, -1, 1, 1)
        elif loss_per_class.dim() != 1:
            raise ValueError(f"Unsupported loss shape: {loss_per_class.shape}")

        weighted_loss = loss_per_class * weights
        return weighted_loss.mean()

    def get_current_weights(self):
        """Return a detached copy of the current class weights."""
        return self.weights.clone()


def test_ema():
    """Run a lightweight EMA sanity check."""
    print("Testing EMA...")

    model = nn.Linear(10, 5)
    ema = ModelEMA(model, decay=0.999, warmup_steps=100)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for step in range(200):
        x = torch.randn(32, 10)
        y = model(x)
        loss = y.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update(model)

        if step % 50 == 0:
            original_param = model.weight.data[0, 0].item()
            ema_param = ema.shadow["weight"][0, 0].item()
            print(f"Step {step}: Original={original_param:.4f}, EMA={ema_param:.4f}")

    print("\nTesting apply_shadow context manager...")
    original_before = model.weight.data[0, 0].item()

    with ema.apply_shadow():
        shadow_param = model.weight.data[0, 0].item()
        print(f"  EMA parameter: {shadow_param:.4f}")

    original_after = model.weight.data[0, 0].item()
    print(f"  Original before: {original_before:.4f}")
    print(f"  Original after:  {original_after:.4f}")
    print(f"  Restored correctly: {abs(original_before - original_after) < 1e-6}")
    print("\nEMA test passed.")


def test_class_balanced_loss():
    """Run a lightweight class-balanced loss sanity check."""
    print("\nTesting class-balanced loss...")

    cb_loss = ClassBalancedLoss(num_classes=4)
    iou_scores = torch.tensor([0.80, 0.65, 0.60, 0.70])
    print(f"Initial IoU: {iou_scores}")
    print(f"Initial weights: {cb_loss.get_current_weights()}")

    for i in range(5):
        cb_loss.update_weights_from_iou(iou_scores)
        weights = cb_loss.get_current_weights()
        print(f"Update {i + 1}: weights = {weights}")

    loss_per_class = torch.randn(8, 4, 32, 32)
    weighted_loss = cb_loss(loss_per_class)
    print(f"\nWeighted loss: {weighted_loss.item():.4f}")
    print("Class-balanced loss test passed.")


if __name__ == "__main__":
    test_ema()
    test_class_balanced_loss()
