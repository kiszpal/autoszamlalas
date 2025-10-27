import cv2
import math
import numpy as np


class EuclideanDistTracker:
    def __init__(self):
        self.center_points = {}
        self.prev_center_points = {}
        self.id_count = 0

    def update(self, objects_rect):
        self.prev_center_points = self.center_points.copy()

        objects_bbs_ids = []

        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            same_object_detected = False

            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])
                if dist < 50:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        self.center_points = new_center_points.copy()
        return objects_bbs_ids


VIDEO_FILE = "video.mp4"
WINDOW_NAME = "Ketiranyu Szamlalo"

cap = cv2.VideoCapture(VIDEO_FILE)
if not cap.isOpened():
    print(f"Hiba: A '{VIDEO_FILE}' nem talalhato vagy nem sikerult megnyitni.")
    exit()

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)

tracker = EuclideanDistTracker()

down_counter = 0
up_counter = 0

counted_ids = []
MIN_AREA = 400  # Minimális terület a detekcióhoz
LINE_Y = 550

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

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

    boxes_ids = tracker.update(detections)

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
