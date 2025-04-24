import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, AutoImageProcessor
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import numpy as np
import json
import argparse
from omegaconf import OmegaConf

name2num_classes = {
    'MNIST': 10,
    'FashionMNIST': 10,
    'CIFAR10': 10,
    'CIFAR100': 100,
}

image_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224', use_fast=True)

class ImageProcessorTransform:
    def __init__(self, image_processor):
        self.image_processor = image_processor

    def __call__(self, img):
        return self.image_processor(img, return_tensors='pt')['pixel_values'][0]


transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),
    ImageProcessorTransform(image_processor),
])

name2dataset = {
    'MNIST': datasets.MNIST,
    'FashionMNIST': datasets.FashionMNIST,
    'CIFAR10': datasets.CIFAR10,
    'CIFAR100': datasets.CIFAR100,
}

###############################################################################

def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(device)

    print(torch.version.cuda)  # Должно выводить версию, например, '11.8'
    print(torch.backends.cudnn.version())  # Должно быть целое число, например, 8500

    train_dataset = name2dataset[config.dataset_name](
        f'~/.pytorch/{config.dataset_name}_data/', download=True, train=True, transform=transform
    )
    train_dataset = torch.utils.data.Subset(train_dataset, np.arange(config.train_size // 3))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.train_batch_size, shuffle=True
    )

    if not hasattr(config, 'output_size'):
        config.output_size = name2num_classes[config.dataset_name]

    results = []

    for rank in tqdm(config.rank_list, desc='LoRA rank loop'):
        model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            num_labels=config.output_size,
            ignore_mismatched_sizes=True
        ).to(device)

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(model, lora_config)

        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        print(config.num_epochs)

        for epoch in tqdm(range(config.num_epochs), desc='Train loop', leave=False):
            model.train()
            for pixel_values, labels in tqdm(train_dataloader, leave=False):
                pixel_values = pixel_values.to(device)
                labels = labels.to(device)
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        model.eval()
        loss_values = []
        with torch.no_grad():
            for pixel_values, label in tqdm(train_dataset, desc='Loss values loop', leave=False):
                pixel_values = pixel_values.to(device)
                labels = torch.tensor([label]).to(device)
                outputs = model(pixel_values=pixel_values.unsqueeze(0), labels=labels)
                loss = outputs.loss.item()
                loss_values.append(loss)

        results.append({'rank': rank, 'loss_values': loss_values})

    with open(config.save_path, 'w') as f:
        json.dump(results, f, indent=4)

###############################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Getting loss values for Transformer with LoRA")
    parser.add_argument("--config_path", type=str, default='config.yml', help="Path to the config file")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the results")
    parser.add_argument("--num_epochs", type=int, required=True, help="Number of training epochs")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    cli_config = OmegaConf.create({
        'save_path': args.save_path,
        'num_epochs': args.num_epochs
    })
    config = OmegaConf.merge(config, cli_config)

    main(config)