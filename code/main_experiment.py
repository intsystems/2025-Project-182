from omegaconf import OmegaConf
from argparse import ArgumentParser

# from src.experiments.vit_classification.assumption_1 import main
from src.experiments.vit_classification.loss_landscape_convergence import main


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)
    main(conf)
