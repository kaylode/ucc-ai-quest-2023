from ultralytics import YOLO
import argparse
import os

os.environ['WANDB_DISABLED'] = 'true'

parser = argparse.ArgumentParser()
parser.add_argument("--model",'-n', type=str, default='yolov8n')
parser.add_argument("--config-file",'-cfg', type=str, default=8)
parser.add_argument("--img_size",'-sz', type=int, default=480)
parser.add_argument("--phase", type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    model = YOLO(f"{args.model}-seg.pt")
    results = model.train(
        batch=16,
        device=0,
        data=args.config_file,
        epochs=50,
        imgsz=args.img_size,
        save=True,
        save_period=1,
        workers=4,
        project=f'runs',
        name=f'{args.phase}/yolov8/{args.model}-seg',
        verbose=True,
        plots=True
    )

    print(results)