# Cyanoclops

A YOLO-based deep learning project for cyanobacteria detection.

## Project Structure

```
cyanoclops/
├── src/
│   ├── train.py          # Training script
│   └── predict.py        # Prediction script
├── data/
│   ├── dataset.yaml      # Dataset configuration
│   ├── images/           # Training images
│   └── labels/           # Annotation labels
└── requirements.txt      # Project dependencies
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the YOLO model:

```bash
python src/train.py --model yolo11m.pt --epochs 200 --data data/dataset.yaml
```

**Options:**
- `--model`: YOLO model to use (default: `yolo11m.pt`)
- `--epochs`: Number of training epochs (default: `200`)
- `--data`: Path to dataset configuration (default: `data/dataset.yaml`)
- `--device`: Device for training, e.g. `0` for GPU or `cpu`; if omitted the script will use GPU if available, otherwise CPU.

### Prediction

Make predictions on new images:

```bash
python src/predict.py --source path/to/image_or_folder
```

## Requirements

See `requirements.txt` for all dependencies.

## License

TBD
