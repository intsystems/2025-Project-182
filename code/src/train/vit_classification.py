import os
import time
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers.optimization import get_constant_schedule_with_warmup

from ..datasets import get_dataloader
from ..models.vit_classification import get_vit
from .checkpoint import load_trainer_checkpoint, save_trainer_checkpoint


def train(conf, model, train_dataloader, val_dataloader, logger, optimizer, scheduler):
    device = torch.device(conf.trainer.device)

    if conf.trainer.checkpoint_path is not None:
        print(f"Loading trainer chekcpoint from {conf.trainer.checkpoint_path}...")
        model, optimizer, scheduler = load_trainer_checkpoint(
            conf, model, optimizer, scheduler
        )
        torch.cuda.empty_cache()

    if conf.metrics.validate_on_start:
        print("Running validation on start...")
        log_loss, log_acc = validate(conf, model, val_dataloader)
        step = scheduler.state_dict()["last_epoch"]
        logger.add_scalar("val/loss", log_loss, step)
        logger.add_scalar("val/accuracy", log_acc, step)
        print(f"[Val] Step {step} | Loss: {log_loss:.4f} | Acc: {log_acc*100:.2f}%")

    step_times = []
    for epoch in range(conf.trainer.num_epochs):
        for x, y in train_dataloader:
            start_time = time.perf_counter()
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=conf.optimizer.max_norm, norm_type=2.0
            )
            optimizer.step()
            scheduler.step()

            step = scheduler.state_dict()["last_epoch"]
            step_times.append(time.perf_counter() - start_time)
            if step % conf.logger.log_interval == 0:
                log_loss = loss.detach()
                step_time = sum(step_times) / len(step_times)
                step_times = []
                logger.add_scalar("train/loss", log_loss, step)
                logger.add_scalar(
                    "train/learning_rate", scheduler.get_last_lr()[0], step
                )
                print(
                    f"Step {step} | Epoch {epoch} | Loss: {log_loss:.4f} | Step Time {step_time:.2f} s"
                )

            for checkpoint_name, checkpoint_type in zip(
                [f"step_{step}", "last"],
                ["regular_save_interval", "last_save_interval"],
            ):
                if step % conf.trainer.checkpoint[checkpoint_type] == 0:
                    conf.trainer.checkpoint_path = os.path.join(
                        conf.trainer.checkpoint.root_dir, checkpoint_name
                    )
                    save_trainer_checkpoint(conf, model, optimizer, scheduler)

            if step % conf.metrics.validation_interval == 0:
                log_loss, log_acc = validate(conf, model, val_dataloader)
                logger.add_scalar("val/loss", log_loss, step)
                logger.add_scalar("val/accuracy", log_acc, step)
                print(
                    f"[Val] Step {step} | Epoch {epoch} | Loss: {log_loss:.4f} | Acc: {log_acc*100:.2f}%"
                )


def validate(conf, model, val_dataloader):
    device = torch.device(conf.trainer.device)
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in val_dataloader:
            x = x.to(device)
            y = y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            total_loss += loss.item() * y.shape[0]
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.shape[0]
    model.train()
    mean_loss = total_loss / max(total, 1)
    acc = total_correct / max(total, 1)
    return mean_loss, acc


def main(conf):
    torch.set_float32_matmul_precision("medium")
    device = torch.device(conf.trainer.device)
    dtype = torch.bfloat16

    model = get_vit(conf.model.params)
    model.to(device, dtype)

    if conf.model.checkpoint_path is not None and conf.trainer.checkpoint_path is None:
        print(f"Loading checkpoint from: {conf.model.checkpoint_path}")
        state_dict = torch.load(
            conf.model.checkpoint_path, weights_only=True, map_location="cpu"
        )
        model.load_state_dict(state_dict)

    model_size = f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M"
    print("Model size:", model_size)

    logger = SummaryWriter(**conf.logger.tensorboard)
    train_dataloader = get_dataloader(conf.data, train=True)
    val_dataloader = get_dataloader(conf.data, train=False)
    optimizer = torch.optim.AdamW(model.parameters(), **conf.optimizer.params)
    scheduler = get_constant_schedule_with_warmup(optimizer, **conf.scheduler.params)

    train(conf, model, train_dataloader, val_dataloader, logger, optimizer, scheduler)
