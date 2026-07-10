import argparse
from ultralytics import YOLO

def main():
    # Step 1: Parse arguments for prediction
    print("Step 1/3: Parsing prediction arguments...")
    parser = argparse.ArgumentParser(description="Run inference using trained YOLO model.")
    parser.add_argument("--model", type=str, default="runs/detect/train/weights/best.pt", help="Path to trained model weights")
    parser.add_argument("--source", type=str, required=True, help="Path to image or folder for prediction")
    args = parser.parse_args()

    # Step 2: Load the trained model weights
    print(f"Step 2/3: Loading trained model from '{args.model}'...")
    model = YOLO(args.model)

    # Step 3: Run inference and save results
    print(f"Step 3/3: Running prediction on '{args.source}'...")
    model.predict(
        source=args.source,
        save=True,      # Saves the visual results with bounding boxes
        imgsz=640
    )
    
    print("Process finished: Predictions completed and saved successfully!")

if __name__ == "__main__":
    main()