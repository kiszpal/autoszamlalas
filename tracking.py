import argparse
import cv2
import numpy as np
from DistanceTracking import EuclideanDistTracker
import torch

def parse_arguments():
    parser = argparse.ArgumentParser(description="Tracking Application")

    parser.add_argument(
        "--ai",
        type=bool,
        help="Use AI-based tracking if set to True, otherwise use traditional tracking.",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    
    parser.add_argument(
        "--display",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="Display the tracking window if set to True.",
        default=True,
    )
    
    parser.add_argument(
        "--input",
        type=str,
        help="Input video file name.",
        default="video.mp4",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output video file name.",
        default=None,
    )

    return parser.parse_args()

TRADITIONAL_TRACKING = parse_arguments().ai == False


# --- Fő program ---
VIDEO_FILE = "video.mp4"
WINDOW_NAME = "Ketiranyu Szamlalo"

cap = cv2.VideoCapture(VIDEO_FILE)
if not cap.isOpened():
    print(f"Hiba: A '{VIDEO_FILE}' nem talalhato vagy nem sikerult megnyitni.")
    exit()

# -- Tradícionális objektumdetektálás inicializálása --

if TRADITIONAL_TRACKING:
    # Objektum detektor (háttérkivonás)
    object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)
    MIN_AREA = 400  # Minimális terület a detekcióhoz

# -- Vége --

# -- AI alapú objektumdetektálás inicializálása --

if not TRADITIONAL_TRACKING:
    from ultralytics import YOLO
    # AI modell betöltése (pl. YOLOv8)
    model = YOLO("./weights/yolo11n_5p5p5ep.pt")  # Használhat más modellt is
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.to(DEVICE)
    

# -- Vége --

# Követő inicializálása
tracker = EuclideanDistTracker()

down_counter = 0
up_counter = 0

counted_ids = []
LINE_Y = 550  # A számláló vonal Y pozíciója

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    if TRADITIONAL_TRACKING: # Hagyományos objektumdetektálás
        # ROI
        roi_points = np.array([[0, 400], [width, 400], [width, 700], [0, 700]], np.int32)
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, [roi_points], (255, 255, 255))
        roi_frame = cv2.bitwise_and(frame, mask)

        # Maszk az előtér detektálásához
        fg_mask = object_detector.apply(roi_frame)
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)

        # Morfológiai műveletek a zaj csökkentésére
        kernel_erode = np.ones((3, 3), np.uint8)
        fg_mask = cv2.erode(fg_mask, kernel_erode, iterations=2)
        kernel_close = np.ones((11, 11), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)

        # Kontúrok keresése a maszkon
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > MIN_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append([x, y, w, h])
    else: # AI alapú objektumdetektálás
        results = model(frame)
        
        result = []
        names = []
        confs = []
        
        for res in results:
            result = res.boxes.xywh.cpu().numpy().astype(int).tolist()
            # Convert from xywh (center) to x_topleft, y_topleft, width, height
            for i, (xc, yc, w, h) in enumerate(result):
                x_tl = int(xc - w / 2)
                y_tl = int(yc - h / 2)
                x_tl = max(0, x_tl)
                y_tl = max(0, y_tl)
                result[i] = [x_tl, int(y_tl), int(w), int(h)]
            names = [res.names[cls.item()] for cls in res.boxes.cls.int()]
            confs = res.boxes.conf.cpu().numpy().astype(float).tolist()  # confidence score of each box
        
        idx_list_to_delete = set()
        
        for idx, (name, conf) in enumerate(zip(names, confs)):
            print(name, conf, sep=" : ")
            
            if name == "car" or name == "truck" or name == "bus" or name == "motorbike" or name == "vechicle":
                #if conf < 0.6:
                    #idx_list_to_delete.add(idx)
                continue
            else:
                idx_list_to_delete.add(idx)
        
        
        for index in sorted(idx_list_to_delete, reverse=True):
            del result[index]
            del names[index]
            del confs[index]
            
        with open("detections.txt", "a") as f:
            f.write(f'{result}\n')
            f.write(f'{"*"*100}\n')
            f.write(f'{names}\n')
            f.write(f'{"*"*100}\n')
            f.write(f'{confs}\n')
            f.write(f'\n{"-"*100}\n\n')
            
        detections = result
        
    boxes_ids = tracker.update(detections)

    # Számláló vonal kirajzolása
    cv2.line(frame, (0, LINE_Y), (width, LINE_Y), (0, 255, 0), 2)

    for box_id in boxes_ids:
        x, y, w, h, id = box_id

        if id not in counted_ids:
            if id in tracker.prev_center_points:
                prev_y = tracker.prev_center_points[id][1]
                current_y = tracker.center_points[id][1]

                if prev_y < LINE_Y and current_y >= LINE_Y:
                    down_counter += 1
                    counted_ids.append(id)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

                elif prev_y > LINE_Y and current_y <= LINE_Y:
                    up_counter += 1
                    counted_ids.append(id)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)

        cv2.putText(frame, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(frame, "Lefele: " + str(down_counter), (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
    cv2.putText(frame, "Felfele: " + str(up_counter), (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(20)
    if key == 27:
        break

    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()