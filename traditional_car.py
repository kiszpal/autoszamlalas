import cv2
import math
import numpy as np
import customtkinter as ctk
import threading
from PIL import Image
from tkinter import filedialog


# --------------------------
# OpenCV - Tracker Class
# --------------------------
class EuclideanDistTracker:
    """
        Objektumkövető osztály, amely a centrumpontok euklideszi távolsága
        alapján tartja nyilván az objektumok ID-jét a képkockák között.
    """
    def __init__(self):
        """
                Inicializálja a követő rendszert.
                - center_points: Az aktuálisan észlelt objektumok utolsó ismert centrumpontjai és ID-i.
                - prev_center_points: Az előző ciklusban észlelt centrumpontok.
                - id_count: Következő szabad objektum ID.
                - max_distance: Maximális távolság (pixelben), amin belül két pontot azonos objektumnak tekint.
        """
        self.center_points = {}
        self.prev_center_points = {}
        self.id_count = 0
        self.max_distance = 50

    def set_max_distance(self, distance):
        """ Beállítja a maximális távolságot (max_distance) a követéshez. """
        try:
            self.max_distance = int(distance)
        except ValueError:
            self.max_distance = 50

    def update(self, objects_rect):
        """
                Frissíti az objektumok helyzetét a detektált téglalapok (rect) alapján.
                Végigmegy az új detektálásokon, megkeresi a legközelebbi meglévő objektumot
                a center_points listában (ha a távolság kisebb, mint max_distance).
                Ha talál, frissíti a pozíciót, ha nem, új ID-t rendel az objektumhoz.
        """
        self.prev_center_points = self.center_points.copy()
        objects_bbs_ids = []

        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])
                if dist < self.max_distance:
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


# --------------------------
# CTk GUI
# --------------------------
class TrafficCounterApp(ctk.CTk):
    """
        Fő alkalmazásosztály, amely a grafikus felhasználói felületet (GUI)
        és a videófeldolgozás fő logikáját (threading) kezeli.
    """
    def __init__(self, default_video_file="video.mp4"):
        """
            Inicializálja az alkalmazást, beállítja az alapértelmezett értékeket
            a GUI-hoz és a videófeldolgozáshoz.
        """
        super().__init__()

        self.title("Paraméterezhető - Vezérlőpult")
        self.geometry("1400x900")
        self.video_file = default_video_file

        # --- Paraméterek Alapértelmezett Értékek ---
        self.is_running = False
        self.is_paused = False
        self.show_mask_view = False
        self.LINE_Y_DEFAULT = 450
        self.MIN_AREA_DEFAULT = 200
        self.TARGET_WIDTH = 1280
        self.TARGET_HEIGHT = 720
        self.FPS_DELAY = 33

        # ALAPÉRTELMEZETT ÉRTÉKEK MOG2
        self.MOG2_HISTORY_DEFAULT = 100
        self.MOG2_THRESHOLD_DEFAULT = 50
        self.MORPH_ITERATIONS_OPEN_DEFAULT = 1
        self.MORPH_ITERATIONS_CLOSE_DEFAULT = 2
        self.TRACKER_DISTANCE_DEFAULT = 50

        self.stop_event = threading.Event()
        self.preload_frame = None
        self.processed_frame = None

        # --- Vonal változók ---
        self.down_counter = 0
        self.up_counter = 0
        self.line_y = self.LINE_Y_DEFAULT

        # CTk változók
        self.line_y_var = ctk.DoubleVar(value=self.LINE_Y_DEFAULT)
        self.down_label_var = ctk.StringVar(value=f"Lefelé: {self.down_counter}")
        self.up_label_var = ctk.StringVar(value=f"Felfelé: {self.up_counter}")
        self.status_label_var = ctk.StringVar(value="Állapot: Készenlétben")
        self.min_area_var = ctk.StringVar(value=str(self.MIN_AREA_DEFAULT))
        self.file_path_var = ctk.StringVar(value=self.video_file)
        self.current_tk_image = None

        # CTk változók a paraméterezéshez
        self.mog2_history_var = ctk.StringVar(value=str(self.MOG2_HISTORY_DEFAULT))
        self.mog2_threshold_var = ctk.StringVar(value=str(self.MOG2_THRESHOLD_DEFAULT))
        self.morph_open_var = ctk.StringVar(value=str(self.MORPH_ITERATIONS_OPEN_DEFAULT))
        self.morph_close_var = ctk.StringVar(value=str(self.MORPH_ITERATIONS_CLOSE_DEFAULT))
        self.tracker_distance_var = ctk.DoubleVar(value=self.TRACKER_DISTANCE_DEFAULT)

        self.setup_ui()

        # --- OpenCV / Threading ---
        self.cap = None
        self.video_thread = None

        self.preload_video()

    def setup_ui(self):
        """
            Létrehozza és elrendezi a grafikus elemeket (gombok, csúszkák, videó ablak)
            a CustomTkinter felületen.
        """
        top_frame = ctk.CTkFrame(self, fg_color="transparent")
        top_frame.pack(padx=20, pady=10, fill="x")

        ctk.CTkButton(top_frame, text="Videó tallózása", command=self.browse_video).pack(side="left", padx=10)
        ctk.CTkEntry(top_frame, textvariable=self.file_path_var, width=250).pack(side="left", padx=5)

        # Eredmény és státusz címkék
        ctk.CTkLabel(top_frame, textvariable=self.down_label_var, font=ctk.CTkFont(size=20, weight="bold"),
                     text_color="#0000FF").pack(side="left", padx=10)
        ctk.CTkLabel(top_frame, textvariable=self.up_label_var, font=ctk.CTkFont(size=20, weight="bold"),
                     text_color="#FFA500").pack(side="left", padx=10)

        ctk.CTkButton(top_frame, text="Play / Pause", command=self.toggle_play_pause).pack(side="left", padx=15)
        ctk.CTkButton(top_frame, text="Reset", command=self.reset_video).pack(side="left", padx=5)
        ctk.CTkButton(top_frame, text="Nézet váltás", command=self.toggle_view).pack(side="left", padx=5)

        ctk.CTkLabel(top_frame, textvariable=self.status_label_var, text_color="gray").pack(side="right", padx=10)

        # --- Paraméterek Panel 1: LINE_Y és MIN_AREA ---
        param_frame_1 = ctk.CTkFrame(self, fg_color="gray20")
        param_frame_1.pack(padx=20, pady=5, fill="x")

        # Vonal Magasság Slider
        ctk.CTkLabel(param_frame_1, text="LINE_Y:").pack(side="left", padx=(30, 0))
        self.line_y_slider = ctk.CTkSlider(param_frame_1, from_=50, to=self.TARGET_HEIGHT - 50,
                                           variable=self.line_y_var,
                                           number_of_steps=self.TARGET_HEIGHT // 10,
                                           command=self.update_line_y_from_slider, width=200)
        self.line_y_slider.pack(side="left", padx=10)
        self.line_y_label = ctk.CTkLabel(param_frame_1, text=f"{self.LINE_Y_DEFAULT}")
        self.line_y_label.pack(side="left")

        # Minimális Terület bevitele
        ctk.CTkLabel(param_frame_1, text="MIN_AREA (px²):").pack(side="left", padx=(30, 0))
        self.min_area_entry = ctk.CTkEntry(param_frame_1, textvariable=self.min_area_var, width=80)
        self.min_area_entry.pack(side="left", padx=10)

        # --- Paraméterek Panel 2: MOG2, Morfológia, Tracker Távolság ---
        param_frame_2 = ctk.CTkFrame(self, fg_color="gray20")
        param_frame_2.pack(padx=20, pady=5, fill="x")

        # 1. MOG2 Beállítások
        mog2_frame = ctk.CTkFrame(param_frame_2, fg_color="transparent")
        mog2_frame.pack(side="left", padx=10, pady=5)

        ctk.CTkLabel(mog2_frame, text="MOG2 (Újraindítás kell):", font=ctk.CTkFont(weight="bold")).pack(side="left",
                                                                                                        padx=5)

        ctk.CTkLabel(mog2_frame, text="Hist:").pack(side="left", padx=(10, 0))
        ctk.CTkEntry(mog2_frame, textvariable=self.mog2_history_var, width=50).pack(side="left", padx=5)

        ctk.CTkLabel(mog2_frame, text="Thresh:").pack(side="left", padx=(10, 0))
        ctk.CTkEntry(mog2_frame, textvariable=self.mog2_threshold_var, width=50).pack(side="left", padx=5)

        # 2. Morfológiai Beállítások
        morph_frame = ctk.CTkFrame(param_frame_2, fg_color="transparent")
        morph_frame.pack(side="left", padx=10, pady=5)

        ctk.CTkLabel(morph_frame, text="Morfo It:", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)

        ctk.CTkLabel(morph_frame, text="Open:").pack(side="left", padx=(10, 0))
        ctk.CTkEntry(morph_frame, textvariable=self.morph_open_var, width=40).pack(side="left", padx=5)

        ctk.CTkLabel(morph_frame, text="Close:").pack(side="left", padx=(10, 0))
        ctk.CTkEntry(morph_frame, textvariable=self.morph_close_var, width=40).pack(side="left", padx=5)

        # 3. Tracker Távolság
        tracker_frame = ctk.CTkFrame(param_frame_2, fg_color="transparent")
        tracker_frame.pack(side="left", padx=10, pady=5)

        ctk.CTkLabel(tracker_frame, text="Tracker Dist:").pack(side="left", padx=(10, 0))
        tracker_slider = ctk.CTkSlider(tracker_frame, from_=10, to=200, variable=self.tracker_distance_var,
                                       number_of_steps=190, width=150)
        tracker_slider.pack(side="left", padx=5)
        ctk.CTkLabel(tracker_frame, textvariable=self.tracker_distance_var).pack(side="left")

        # Videó megjelenítő
        self.video_label = ctk.CTkLabel(self, text="Videó Betöltése...",
                                        width=self.TARGET_WIDTH, height=self.TARGET_HEIGHT,
                                        text_color="white", fg_color="black")
        self.video_label.pack(pady=10)

    def browse_video(self):
        file_path = filedialog.askopenfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        if file_path:
            self.stop_processing()
            self.video_file = file_path
            self.file_path_var.set(file_path)
            self.reset_gui_state()
            self.preload_video()

    def update_line_y_from_slider(self, value):
        """
            Megnyit egy fájlválasztó ablakot, lehetővé téve a felhasználónak,
            hogy új videófájlt válasszon. Leállítja az aktuális feldolgozást,
            frissíti az elérési utat, reseteli az állapotot és előre betölti
            az első képkockát.
        """
        self.line_y = int(value)
        self.line_y_label.configure(text=f"{self.line_y}")
        if not self.is_running and self.preload_frame is not None:
            self.display_frame(self.preload_frame.copy())

    def toggle_view(self):
        """ Nézet váltása """
        self.show_mask_view = not self.show_mask_view

    def toggle_play_pause(self):
        """ Média vezérlő gombok """
        if not self.is_running:
            self.is_running = True
            self.is_paused = False
            self.status_label_var.set("Állapot: FUT")
            self.stop_event.clear()
            self.video_thread = threading.Thread(target=self.run_video_processing)
            self.video_thread.start()
            self.after(self.FPS_DELAY, self.update_display)
        elif self.is_paused:
            self.is_paused = False
            self.status_label_var.set("Állapot: FUT")
            self.after(self.FPS_DELAY, self.update_display)
        else:
            self.is_paused = True
            self.status_label_var.set("Állapot: SZÜNETELTETVE")

    def reset_gui_state(self):
        """ GUI és paraméterek visszaállítása"""
        self.down_counter = 0
        self.up_counter = 0
        self.down_label_var.set(f"Lefelé: {self.down_counter}")
        self.up_label_var.set(f"Felfelé: {self.up_counter}")
        self.line_y_var.set(self.LINE_Y_DEFAULT)
        self.line_y = self.LINE_Y_DEFAULT
        self.min_area_var.set(str(self.MIN_AREA_DEFAULT))

        self.mog2_history_var.set(str(self.MOG2_HISTORY_DEFAULT))
        self.mog2_threshold_var.set(str(self.MOG2_THRESHOLD_DEFAULT))
        self.morph_open_var.set(str(self.MORPH_ITERATIONS_OPEN_DEFAULT))
        self.morph_close_var.set(str(self.MORPH_ITERATIONS_CLOSE_DEFAULT))
        self.tracker_distance_var.set(self.TRACKER_DISTANCE_DEFAULT)

    def stop_processing(self):
        """ Folyamat leáálítása """
        if self.is_running:
            self.is_running = False
            self.stop_event.set()
            if self.video_thread and self.video_thread.is_alive():
                self.video_thread.join(timeout=0.5)

    def reset_video(self):
        """ Videő újraindítása """
        self.stop_processing()
        self.reset_gui_state()
        self.status_label_var.set("Állapot: Resetelve, Készenlétben")
        self.preload_video()

    def preload_video(self):
        """ Első frame betöltése """
        temp_cap = cv2.VideoCapture(self.video_file)
        if not temp_cap.isOpened():
            self.status_label_var.set(f"Hiba: '{self.video_file}' nem nyitható meg.")
            self.video_label.configure(text=f"HIBA: {self.video_file} nem található.")
            self.preload_frame = None
            return

        ret, frame = temp_cap.read()
        temp_cap.release()

        if ret:
            # Átméretezés más méretű videók esetén
            if frame.shape[0] != self.TARGET_HEIGHT or frame.shape[1] != self.TARGET_WIDTH:
                frame = cv2.resize(frame, (self.TARGET_WIDTH, self.TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)

            self.preload_frame = frame
            self.display_frame(self.preload_frame.copy())
            self.status_label_var.set("Állapot: Készenlétben. Vonal beállítható.")
        else:
            self.status_label_var.set(f"Hiba: Nem sikerült beolvasni az első frame-et.")
            self.preload_frame = None

    def display_frame(self, frame):
        """ Első frame megjelenítése és a tracking határ kirajzolása """
        if frame is None:
            return

        cv2.line(frame, (0, self.line_y), (self.TARGET_WIDTH, self.line_y), (0, 255, 0), 2)

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_image)

        self.current_tk_image = ctk.CTkImage(light_image=pil_img, size=(self.TARGET_WIDTH, self.TARGET_HEIGHT))

        self.video_label.configure(image=self.current_tk_image, text="")

    def update_display(self):
        """ Új frame lekérdezése """
        if self.is_running and not self.is_paused and self.processed_frame is not None:
            self.display_frame(self.processed_frame.copy())

            self.after(self.FPS_DELAY, self.update_display)

        elif self.is_running and not self.is_paused:
            self.after(self.FPS_DELAY, self.update_display)

    def run_video_processing(self):
        """
            **Fő feldolgozó szál.**
            Beolvassa a paramétereket, inicializálja az OpenCV objektumokat
            (BackgroundSubtractorMOG2, EuclideanDistTracker), és ciklusban futtatja
             a videó frame-ről frame-re történő feldolgozását (maszkolás, kontúrkeresés,
            objektumkövetés, számlálás és a végeredmény kirajzolása).
        """
        try:
            self.cap = cv2.VideoCapture(self.video_file)
            if not self.cap.isOpened():
                self.status_label_var.set(f"Hiba: '{self.video_file}' nem nyitható meg.")
                self.is_running = False
                return

            try:
                # Háttérkivonó paraméterek
                safe_mog2_history = int(
                    float(self.mog2_history_var.get())) if self.mog2_history_var.get() else self.MOG2_HISTORY_DEFAULT
                safe_mog2_threshold = int(float(
                    self.mog2_threshold_var.get())) if self.mog2_threshold_var.get() else self.MOG2_THRESHOLD_DEFAULT

                # Morfológiai iterációk
                safe_morph_open = int(float(
                    self.morph_open_var.get())) if self.morph_open_var.get() else self.MORPH_ITERATIONS_OPEN_DEFAULT
                safe_morph_close = int(float(
                    self.morph_close_var.get())) if self.morph_close_var.get() else self.MORPH_ITERATIONS_CLOSE_DEFAULT

                # Tracker távolság
                safe_tracker_distance = int(self.tracker_distance_var.get())

                # MIN_AREA alapértelmezett
                safe_min_area = int(
                    float(self.min_area_var.get())) if self.min_area_var.get() else self.MIN_AREA_DEFAULT

            except ValueError:
                safe_mog2_history = self.MOG2_HISTORY_DEFAULT
                safe_mog2_threshold = self.MOG2_THRESHOLD_DEFAULT
                safe_morph_open = self.MORPH_ITERATIONS_OPEN_DEFAULT
                safe_morph_close = self.MORPH_ITERATIONS_CLOSE_DEFAULT
                safe_tracker_distance = self.TRACKER_DISTANCE_DEFAULT
                safe_min_area = self.MIN_AREA_DEFAULT

            # --- OpenCV Objektumok létrehozása a paraméterekkel ---
            object_detector = cv2.createBackgroundSubtractorMOG2(
                history=safe_mog2_history,
                varThreshold=safe_mog2_threshold,
            )
            tracker = EuclideanDistTracker()
            tracker.set_max_distance(safe_tracker_distance)

            counted_ids = []

            while self.is_running and self.cap.isOpened() and not self.stop_event.is_set():
                if self.is_paused:
                    self.stop_event.wait(1 / 30)
                    continue

                try:
                    MIN_AREA_str = self.min_area_var.get()
                    if MIN_AREA_str:
                        MIN_AREA = int(float(MIN_AREA_str))
                    else:
                        MIN_AREA = safe_min_area

                    tracker_dist_runtime = int(self.tracker_distance_var.get())
                    tracker.set_max_distance(tracker_dist_runtime)

                except ValueError:
                    MIN_AREA = safe_min_area  # Marad a ciklus eleji érték, ha hibás bevitelt kap

                MORPH_OPEN_IT = safe_morph_open
                MORPH_CLOSE_IT = safe_morph_close

                ret, frame = self.cap.read()
                if not ret:
                    break

                if frame.shape[0] != self.TARGET_HEIGHT or frame.shape[1] != self.TARGET_WIDTH:
                    frame = cv2.resize(frame, (self.TARGET_WIDTH, self.TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)

                height, width, _ = frame.shape
                LINE_Y = self.line_y

                roi_points = np.array([[0, 300], [width, 300], [width, 600], [0, 600]], np.int32)
                mask_roi = np.zeros_like(frame)
                cv2.fillPoly(mask_roi, [roi_points], (255, 255, 255))
                roi_frame = cv2.bitwise_and(frame, mask_roi)

                fg_mask = object_detector.apply(roi_frame)
                _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)

                kernel_open = np.ones((3, 3), np.uint8)
                kernel_close = np.ones((7, 7), np.uint8)
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open, iterations=MORPH_OPEN_IT)
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close, iterations=MORPH_CLOSE_IT)

                mask_view_frame = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

                contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                detections = []
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > MIN_AREA:
                        x, y, w, h = cv2.boundingRect(cnt)
                        detections.append([x, y, w, h])

                boxes_ids = tracker.update(detections)

                current_frame = (mask_view_frame if self.show_mask_view else frame).copy()

                for box_id in boxes_ids:
                    x, y, w, h, id = box_id

                    if id not in counted_ids:
                        if id in tracker.prev_center_points and id in tracker.center_points:
                            prev_y = tracker.prev_center_points[id][1]
                            current_y = tracker.center_points[id][1]

                            if prev_y < LINE_Y and current_y >= LINE_Y:
                                self.down_counter += 1
                                counted_ids.append(id)
                                cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                            elif prev_y > LINE_Y and current_y <= LINE_Y:
                                self.up_counter += 1
                                counted_ids.append(id)
                                cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 255), 3)

                    cv2.putText(current_frame, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
                    cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                self.down_label_var.set(f"Lefelé: {self.down_counter}")
                self.up_label_var.set(f"Felfelé: {self.up_counter}")

                self.processed_frame = current_frame

        except Exception as e:
            print(f"Hiba a videófeldolgozásban: {e}")
            self.after(0, lambda: self.status_label_var.set(f"Hiba: {e}"))
        finally:
            if self.cap:
                self.cap.release()

            if not self.stop_event.is_set():
                self.after(0, lambda: self.status_label_var.set("Állapot: Befejeződött"))
                self.is_running = False
            self.stop_event.clear()


if __name__ == "__main__":
    """
        Az alkalmazás indítási pontja. Létrehozza a TrafficCounterApp példányát és elindítja a fő ciklust.
    """
    app = TrafficCounterApp(default_video_file="video.mp4")
    app.mainloop()