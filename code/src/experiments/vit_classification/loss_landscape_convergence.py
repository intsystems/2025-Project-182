import os
import torch
import pandas as pd
import torch.nn.functional as F

from ...datasets import get_dataloader
from ...models.vit_classification import get_vit


def experiment(conf, model, train_dataloader):
    device = torch.device(conf.trainer.device)
    dtype = torch.bfloat16  # под autocast ниже

    model.eval()
    # фиксируем точку w* — НЕ обновляем параметры
    for p in model.parameters():
        p.requires_grad_(False)

    # накапливаемая сумма потерь и число примеров
    S = 0.0  # суммарная loss (по всем примерам)
    N = 0  # суммарное число примеров

    rows = []
    step = 0
    log_interval = conf.experiment.log_interval
    save_path = conf.experiment.save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(conf.trainer.num_epochs):
        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)
            bs = y.shape[0]

            # считаем суммарную батчевую потерю на w*
            with torch.no_grad() and torch.autocast(
                device_type="cuda", dtype=dtype, enabled=("cuda" in device.type)
            ):
                logits = model(x)
                loss_sum_batch = F.cross_entropy(logits, y, reduction="sum").float()

            L_k = (S / N) if N > 0 else float("nan")  # для первого шага нет L_k
            # обновляем накопители, получаем L_{k+1}
            S += loss_sum_batch.item()
            N += bs
            L_k1 = S / max(N, 1)

            delta = abs(L_k1 - L_k) if N > bs else float("nan")

            rows.append(
                {
                    "step": step,
                    "epoch": epoch,
                    "batch_size": bs,
                    "num_samples": N,
                    "L_k": L_k,
                    "L_k1": L_k1,
                    "delta": delta,
                }
            )

            if step % log_interval == 0:
                print(
                    f"step {step:5d} | epoch {epoch:2d} | N={N:7d} | "
                    f"L_k={L_k} → L_k+1={L_k1} | Δ={delta}"
                )
                pd.DataFrame(rows).to_csv(save_path, index=False)

            step += 1

    # финальная запись
    pd.DataFrame(rows).to_csv(save_path, index=False)
    print(f"[OK] saved CSV to {save_path}")


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
