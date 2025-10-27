# ...existing code...
'''
A tanítási dataset linkje: https://www.kaggle.com/datasets/mbsoroush/car-camera-photos/data

A dataset yolo formátumba történő átalakításában a következő notebook segített: https://www.kaggle.com/code/hassanfarid004/yolov8-cars-detection
'''
from multiprocessing import freeze_support

def main():
    try:
        from ultralytics import YOLO
        import torch
    except OSError as e:
        print("Hiba: nem sikerült importálni a torch/ultralytics csomagokat:")
        print(e)
        return

    
    try:
        model = YOLO('yolo11n.pt')
    except Exception as e:
        print("Model betöltési hiba:", e)
        return

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    model.to(DEVICE)

    try:
        results = model.train(
            data='./data/trafficcam/data.yaml',
            epochs=5,
            imgsz=640,
            batch=2,
            workers=0,  
        )
        print(results)
    except Exception as e:
        print("Training hiba:")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    freeze_support()
    main()