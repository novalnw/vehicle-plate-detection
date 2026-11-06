import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train a YOLO plate detector")
    parser.add_argument(
        "--data",
        type=str,
        default="dataset/indonesian-plate-yolo/data.yaml",
        help="Path to data.yaml",
    )
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base weights")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--name", type=str, default="plate_model")
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")

    model = YOLO(args.model)
    model.train(data=str(data_path), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, name=args.name)


if __name__ == "__main__":
    main()
