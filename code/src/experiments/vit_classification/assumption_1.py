import os
import torch
import pandas as pd
import torch.nn.functional as F

from ...datasets import get_dataloader
from ...models.vit_classification import get_vit


def experiment(conf, model, train_dataloader):
    device = torch.device(conf.trainer.device)
    model.eval()
    model.zero_grad(set_to_none=True)

    seen = 0
    grad_sum = [torch.zeros_like(p) for p in model.parameters() if p.requires_grad]
    out = []

    step = 0

    for epoch in range(conf.trainer.num_epochs):
        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(
                    logits, y, reduction="sum"
                )  # sum, will divide later

            model.zero_grad(set_to_none=True)
            loss.backward()

            # accumulate gradiens sum, will average later
            idx = 0
            for p in model.parameters():
                if not p.requires_grad:
                    continue
                if p.grad is None:
                    idx += 1
                    continue
                grad_sum[idx].add_(p.grad.detach())
                idx += 1

            # average gradients over seen data
            seen += y.shape[0]

            if step % 16 == 0:
                flat = torch.cat([(g / seen).reshape(-1) for g in grad_sum])
                out.append((seen, flat.norm(p=2).item()))
                df = pd.DataFrame(out, columns=["num_samples", "grad_norm"])
                os.makedirs(os.path.dirname(conf.experiment.save_path), exist_ok=True)
                df.to_csv(conf.experiment.save_path, index=False)
                print(
                    f"Step {step} | Epoch {epoch} | Grad norm {flat.norm(p=2).item()}"
                )

            step += 1


def main(conf):
    torch.set_float32_matmul_precision("medium")
    device = torch.device(conf.trainer.device)
    dtype = torch.bfloat16

    model = get_vit(conf.model.params)
    model.to(device, dtype)

    print(f"Loading checkpoint from: {conf.model.checkpoint_path}")
    state_dict = torch.load(
        conf.model.checkpoint_path, weights_only=True, map_location="cpu"
    )
    model.load_state_dict(state_dict)

    model_size = f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M"
    print("Model size:", model_size)

    train_dataloader = get_dataloader(conf.data, train=True)

    experiment(conf, model, train_dataloader)
