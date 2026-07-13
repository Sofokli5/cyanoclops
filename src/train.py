import argparse
from ultralytics import YOLO
import torch

def main():
    # Step 1: Parse command line arguments
    print("Step 1/3: Parsing configuration arguments...")
    parser = argparse.ArgumentParser(description="Train YOLO models for Cyanobacteria detection.")
    parser.add_argument("--model", type=str, default="yolo11m.pt", help="YOLO model configuration (e.g., yolo11n.pt, yolo11m.pt)")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--data", type=str, default="data/dataset.yaml", help="Path to dataset.yaml file")
    parser.add_argument("--device", type=str, default=None, help="Device to use for training, e.g. '0' or 'cpu'")
    args = parser.parse_args()

    # Step 2: Load the specified YOLO model
    print(f"Step 2/3: Loading model '{args.model}'...")
    model = YOLO(args.model)

    # Select device automatically if not provided
    if args.device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Step 3/3: Starting training on {args.data} for {args.epochs} epochs using device={device}...")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=640,
        device=device
    )
    
    print("Process finished: Training completed successfully!")

if __name__ == "__main__":
    main()