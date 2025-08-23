import argparse
import wandb
import yaml
from train import train
from configs import settings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id", type=str, default=None, help="Existing W&B sweep ID")
    parser.add_argument("--count", type=int, default=5, help="Number of runs to launch")
    args = parser.parse_args()

    if args.sweep_id:
        sweep_id = args.sweep_id
    else:
        with open(settings.SWEEP_CONFIG_PATH / "sweep_resnet.yaml") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        sweep_id = wandb.sweep(sweep=config, project=settings.WANDB_PROJECT_NAME)

    wandb.agent(sweep_id, function=train, count=args.count)

if __name__ == "__main__":
    main()
