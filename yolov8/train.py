from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model",'-n', type=str, default='yolov8n')
parser.add_argument("--config-file",'-cfg', type=str, default=8)
parser.add_argument("--img_size",'-sz', type=int, default=480)

if __name__ == "__main__":
    args = parser.parse_args()
    model = YOLO(f"{args.model}-seg.pt")
    results = model.train(
        batch=8,
        device=0,
        data=args.config_file,
        epochs=20,
        imgsz=args.img_size,
    )

    print(results)
    import pdb; pdb.set_trace()