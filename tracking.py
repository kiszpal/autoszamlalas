import argparse
import cv2
import numpy as np
from DistanceTracking import EuclideanDistTracker
import torch
from collections import defaultdict


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


def get_color_for_type(vehicle_type, color_map):
    """Generate a unique color for each vehicle type"""
    if vehicle_type not in color_map:
        # Generate a random but consistent color for new types
        np.random.seed(hash(vehicle_type) % (2 ** 32))
        color = tuple(np.random.randint(50, 255, size=3).tolist())
        color_map[vehicle_type] = color
    return color_map[vehicle_type]


args = parse_arguments()
TRADITIONAL_TRACKING = args.ai == False
DISPLAY_TRACKING = args.display == True

# --- Fő program ---
VIDEO_FILE = args.input
WINDOW_NAME = "Ketiranyu Szamlalo"

DEFAULT_COLOR = (0, 255, 0)
vehicle_color_map = {}

cap = cv2.VideoCapture(VIDEO_FILE)
if not cap.isOpened():
    print(f"Hiba: A '{VIDEO_FILE}' nem talalhato vagy nem sikerult megnyitni.")
    exit()

video_writer = None
if args.output:
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))

    if not video_writer.isOpened():
        print(f"Hiba: Nem sikerult letrehozni a kimeneti videot: '{args.output}'")
        video_writer = None
    else:
        print(f"Kimenet mentese: {args.output}")

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
    model = YOLO("./weights/trafic_5.pt")  # Használhat más modellt is
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.to(DEVICE)
    

# -- Vége --

# Követő inicializálása
tracker = EuclideanDistTracker()

down_counter = defaultdict(int)
up_counter = defaultdict(int)
down_counter["total"] = 0
up_counter["total"] = 0

counted_ids = []
LINE_Y = 550  # A számláló vonal Y pozíciója

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    detection_types = []

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
            
            if conf < 0.0: # Kesobbi tanitasok utan novelheto, egyelore tul alacsonyak a konfidenciaszintek
                idx_list_to_delete.add(idx)
                continue
        
        
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
        detection_types = names

    boxes_ids = tracker.update(detections)

    type_mapping = {}
    for idx, box_id in enumerate(boxes_ids):
        if idx < len(detection_types):
            type_mapping[box_id[4]] = detection_types[idx]

    # Számláló vonal kirajzolása
    cv2.line(frame, (0, LINE_Y), (width, LINE_Y), (0, 255, 0), 2)

    for box_id in boxes_ids:
        x, y, w, h, id = box_id

        if not TRADITIONAL_TRACKING and id in type_mapping:
            vehicle_type = type_mapping[id]
            box_color = get_color_for_type(vehicle_type, vehicle_color_map)
        else:
            vehicle_type = "unknown"
            box_color = DEFAULT_COLOR

        if id not in counted_ids:
            if id in tracker.prev_center_points:
                prev_y = tracker.prev_center_points[id][1]
                current_y = tracker.center_points[id][1]

                if prev_y < LINE_Y and current_y >= LINE_Y:
                    # Típus számlálás
                    down_counter[vehicle_type] += 1
                    down_counter["total"] += 1
                    counted_ids.append(id)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

                elif prev_y > LINE_Y and current_y <= LINE_Y:
                    # Típus számlálás
                    up_counter[vehicle_type] += 1
                    up_counter["total"] += 1
                    counted_ids.append(id)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)

        # Típus címke
        if not TRADITIONAL_TRACKING and id in type_mapping:
            label = f"{type_mapping[id]} {id}"
        else:
            label = str(id)

        cv2.putText(frame, label, (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

    # Dinamikus számláló
    down_text = f"Lefele: {down_counter['total']}"
    up_text = f"Felfele: {up_counter['total']}"

    if not TRADITIONAL_TRACKING:
        down_types = [f"{vtype}:{count}" for vtype, count in sorted(down_counter.items()) if vtype != "total"]
        up_types = [f"{vtype}:{count}" for vtype, count in sorted(up_counter.items()) if vtype != "total"]

        if down_types:
            down_text = f"Lefele: {down_counter['total']} ({' '.join(down_types)})"
        if up_types:
            up_text = f"Felfele: {up_counter['total']} ({' '.join(up_types)})"

    cv2.putText(frame, down_text, (50, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
    cv2.putText(frame, up_text, (50, 80), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

    if not TRADITIONAL_TRACKING and vehicle_color_map:
        legend_x = width - 200
        legend_y = 50
        cv2.putText(frame, "Legend:", (legend_x, legend_y), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
        legend_y += 30

        for vtype in sorted(vehicle_color_map.keys()):
            color = vehicle_color_map[vtype]
            cv2.rectangle(frame, (legend_x, legend_y - 15), (legend_x + 20, legend_y), color, -1)
            cv2.putText(frame, vtype.capitalize(), (legend_x + 30, legend_y), cv2.FONT_HERSHEY_PLAIN, 1.2,
                        (255, 255, 255), 2)
            legend_y += 25

    if video_writer is not None:
        video_writer.write(frame)

    if DISPLAY_TRACKING:
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(20)
        if key == 27:
            break

        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break
    else:
        key = cv2.waitKey(1)
        if key == 27:
            break

cap.release()
if video_writer is not None:
    video_writer.release()
    print(f"Kimenet sikeresen mentve: {args.output}")
cv2.destroyAllWindows()