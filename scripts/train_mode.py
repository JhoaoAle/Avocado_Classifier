from ultralytics import YOLO
import yaml

from ultralytics import YOLO

if __name__ == "__main__":
    with open("config/model_config.yml", "r") as f:
        config = yaml.safe_load(f)

        data_path = "config/data_config.yml"

        model = YOLO("runs/segment/train8/weights/best.pt")

        model.train(
            data=data_path,
            epochs=config["epochs"],
            imgsz=config["img_size"],
            batch=config["batch_size"],
            lr0=config["lr0"],
            patience=config["patience"],
            optimizer=config["optimizer"],
            augment=config["augment"]
        )
    



