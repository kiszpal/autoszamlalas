import tkinter as tk
from tkinter import filedialog, scrolledtext
import subprocess
import threading
import sys
import os


class TrackingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Tracking Launcher")
        self.root.geometry("600x500")

        self.pad_x = 10
        self.pad_y = 5

        self.input_path = tk.StringVar(value="video.mp4")
        self.output_path = tk.StringVar(value="")
        self.use_ai = tk.BooleanVar(value=False)
        self.show_display = tk.BooleanVar(value=True)
        self.process = None

        self.create_widgets()

    def create_widgets(self):
        title_label = tk.Label(self.root, text="Traffic Tracking Configuration", font=("Helvetica", 16, "bold"))
        title_label.pack(pady=15)

        form_frame = tk.Frame(self.root)
        form_frame.pack(fill="x", padx=20)

        # 1. Input File
        tk.Label(form_frame, text="Input Video File:").grid(row=0, column=0, sticky="w", pady=self.pad_y)
        tk.Entry(form_frame, textvariable=self.input_path, width=40).grid(row=0, column=1, padx=self.pad_x,
                                                                          pady=self.pad_y)
        tk.Button(form_frame, text="Browse...", command=self.browse_input).grid(row=0, column=2, padx=self.pad_x,
                                                                                pady=self.pad_y)

        # 2. Output File
        tk.Label(form_frame, text="Output Video File (Optional):").grid(row=1, column=0, sticky="w", pady=self.pad_y)
        tk.Entry(form_frame, textvariable=self.output_path, width=40).grid(row=1, column=1, padx=self.pad_x,
                                                                           pady=self.pad_y)
        tk.Button(form_frame, text="Save As...", command=self.browse_output).grid(row=1, column=2, padx=self.pad_x,
                                                                                  pady=self.pad_y)

        # Separator
        tk.Frame(form_frame, height=2, bd=1, relief="sunken").grid(row=2, column=0, columnspan=3, sticky="ew", pady=15)

        # 3. Settings (Checkboxes)
        settings_frame = tk.Frame(form_frame)
        settings_frame.grid(row=3, column=0, columnspan=3, sticky="w")

        # AI Toggle
        tk.Checkbutton(settings_frame, text="Use AI Tracking (YOLO)", variable=self.use_ai,
                       font=("Helvetica", 10, "bold"), fg="#2c3e50").pack(anchor="w", pady=2)
        tk.Label(settings_frame, text="   (Uncheck for Traditional Background Subtraction)",
                 font=("Helvetica", 8), fg="gray").pack(anchor="w")

        # Display Toggle
        tk.Checkbutton(settings_frame, text="Show Live Display Window", variable=self.show_display).pack(anchor="w",
                                                                                                         pady=(10, 2))

        # 4. Action Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=20)

        self.run_btn = tk.Button(btn_frame, text="START TRACKING", bg="#27ae60", fg="white",
                                 font=("Helvetica", 12, "bold"), width=20, height=2, command=self.start_thread)
        self.run_btn.pack(side="left", padx=10)

        self.quit_btn = tk.Button(btn_frame, text="EXIT", bg="#c0392b", fg="white",
                                  font=("Helvetica", 12, "bold"), width=10, height=2, command=self.root.quit)
        self.quit_btn.pack(side="left", padx=10)

        # 5. Log Output
        tk.Label(self.root, text="Console Output:").pack(anchor="w", padx=20)
        self.log_area = scrolledtext.ScrolledText(self.root, height=8, state='disabled', font=("Consolas", 9))
        self.log_area.pack(fill="both", expand=True, padx=20, pady=(0, 20))

    def browse_input(self):
        filename = filedialog.askopenfilename(title="Select Input Video",
                                              filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"),
                                                         ("All files", "*.*")])
        if filename:
            self.input_path.set(filename)

    def browse_output(self):
        filename = filedialog.asksaveasfilename(title="Save Output Video",
                                                defaultextension=".mp4",
                                                filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi")])
        if filename:
            self.output_path.set(filename)

    def log(self, message):
        self.log_area.config(state='normal')
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)
        self.log_area.config(state='disabled')

    def start_thread(self):
        self.run_btn.config(state="disabled", text="RUNNING...")
        self.log("Starting process...")

        threading.Thread(target=self.run_script, daemon=True).start()

    def run_script(self):
        script_name = "tracking.py"

        if not os.path.exists(script_name):
            self.log(f"Error: {script_name} not found in current directory!")
            self.root.after(0, lambda: self.run_btn.config(state="normal", text="START TRACKING"))
            return

        cmd = [sys.executable, script_name]

        cmd.extend(["--input", self.input_path.get()])

        if self.output_path.get().strip():
            cmd.extend(["--output", self.output_path.get()])

        if self.use_ai.get():
            cmd.append("--ai")
        else:
            cmd.append("--no-ai")

        if self.show_display.get():
            cmd.append("--display")
        else:
            cmd.append("--no-display")

        self.log(f"Executing: {' '.join(cmd)}")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    self.log(output.strip())

            stderr = process.communicate()[1]
            if stderr:
                self.log(f"ERRORS:\n{stderr}")

            return_code = process.poll()
            if return_code == 0:
                self.log("Finished successfully.")
            else:
                self.log(f"Finished with exit code {return_code}.")

        except Exception as e:
            self.log(f"Failed to run script: {e}")
        finally:
            self.root.after(0, lambda: self.run_btn.config(state="normal", text="START TRACKING"))


if __name__ == "__main__":
    root = tk.Tk()
    app = TrackingApp(root)
    root.mainloop()