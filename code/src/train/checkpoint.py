import os
import torch
from omegaconf import OmegaConf


def load_trainer_checkpoint(conf, model, optimizer, scheduler):
    save_dir = conf.trainer.checkpoint_path
    model_path = os.path.join(save_dir, "model.pt")
    optimizer_path = os.path.join(save_dir, "optimizer.pt")
    scheduler_path = os.path.join(save_dir, "scheduler.pt")

    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    optimizer.load_state_dict(
        torch.load(optimizer_path, map_location="cpu", weights_only=True)
    )
    scheduler.load_state_dict(
        torch.load(scheduler_path, map_location="cpu", weights_only=False)
    )
    return model, optimizer, scheduler


def save_trainer_checkpoint(conf, model, optimizer, scheduler):
    save_dir = conf.trainer.checkpoint_path
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "model.pt")
    optimizer_path = os.path.join(save_dir, "optimizer.pt")
    scheduler_path = os.path.join(save_dir, "scheduler.pt")

    torch.save(model.state_dict(), model_path)
    torch.save(optimizer.state_dict(), optimizer_path)
    torch.save(scheduler.state_dict(), scheduler_path)
    OmegaConf.save(conf, conf.trainer.conf_path)
