
'''
A tanítási dataset linkje: https://www.kaggle.com/datasets/mbsoroush/car-camera-photos/data

A dataset yolo formátumba történő átalakításában a következő notebook segített: https://www.kaggle.com/code/hassanfarid004/yolov8-cars-detection
'''
from ultralytics import YOLO
import torch
from multiprocessing import freeze_support

def main():
    model = YOLO('./weights/yolo11n_5p5ep.pt')
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    model.to(DEVICE)

    results = model.train(
        data='./data/car_dataset_yolo/data.yaml',
        epochs=5,
        imgsz=640,
        batch=8,  
        workers=2 
    )

    print(results)

if __name__ == '__main__':
    freeze_support()
    main()
