import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import sqlite3
import cv2
import json
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# Paths
MODEL_PATH = "item_classifier_model"
DB_PATH = "collection_catalog.db"
CLASS_NAMES_PATH = "class_names.json"
PHOTO_DIR = "photos"

# Load model and class names
model = load_model(MODEL_PATH)
with open(CLASS_NAMES_PATH, "r") as f:
    CLASS_NAMES = json.load(f)


def apply_tta(img_array, augmentations=5):
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.05,
        height_shift_range=0.05,
        brightness_range=[0.8, 1.2]
    )
    predictions = []
    for _ in range(augmentations):
        it = datagen.flow(img_array, batch_size=1)
        predictions.append(model.predict(it)[0])
    return np.mean(predictions, axis=0)


def predict_with_description(input_data, top_k=1):
    if isinstance(input_data, str):
        img = image.load_img(input_data, target_size=(224, 224))
        x = image.img_to_array(img).astype("float32") / 255.0
    else:
        frame = cv2.resize(input_data, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x = frame.astype("float32") / 255.0

    x = np.expand_dims(x, axis=0)
    predictions = apply_tta(x, augmentations=8)
    top_indices = np.argsort(predictions)[::-1][:top_k]

    results = []
    for idx in top_indices:
        class_name = CLASS_NAMES[idx]
        confidence = predictions[idx]
        db_name = class_name.replace('_', ' ')
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT description, fact FROM items WHERE name = ?", (db_name,))
        row = cursor.fetchone()
        conn.close()

        description = row[0] if row and row[0] else "No description found."
        fact = row[1] if row and row[1] else ""
        results.append((class_name, confidence, description, fact))

    return results


class LibraryWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Item Library")

        self.geometry("1200x700")

        self.left_frame = tk.Frame(self)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        self.listbox = tk.Listbox(self.left_frame, width=40)
        self.listbox.pack(fill=tk.Y, expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.display_item_info)

        self.right_frame = tk.Frame(self)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH,
                              expand=True, padx=5, pady=5)
# === Image canvas and horizontal scrollbar ===
        self.image_frame = tk.Frame(self.right_frame)
        self.image_frame.pack(fill='x', pady=(0, 10))

        self.image_canvas = tk.Canvas(self.image_frame, height=320, bg='white')
        self.image_canvas.pack(side='top', fill='x', expand=True)

        self.scroll_x = tk.Scrollbar(
            self.image_frame, orient='horizontal', command=self.image_canvas.xview)
        self.scroll_x.pack(side='bottom', fill='x')

        self.image_canvas.configure(xscrollcommand=self.scroll_x.set)

        self.image_canvas.bind("<ButtonPress-1>", self.start_scroll)
        self.image_canvas.bind("<B1-Motion>", self.do_scroll)

        self.info_text = tk.Text(self.right_frame, wrap='word')
        self.info_text.pack(fill=tk.BOTH, expand=True)

        self.photo_refs = []

        self.load_items()

    def start_scroll(self, event):
        self.image_canvas.scan_mark(event.x, event.y)

    def do_scroll(self, event):
        self.image_canvas.scan_dragto(event.x, event.y, gain=1)

    def load_items(self):
        self.title("Item Library")
        self.left_frame = tk.Frame(self.master)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.right_frame = tk.Frame(self.master)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.result_text = tk.Text(self.right_frame, height=10, wrap=tk.WORD)
        self.result_text.pack(fill=tk.X, pady=(10, 0))
        self.result_text.insert(tk.END, "Model not yet loaded.\n")
        self.result_text.config(state='disabled')

        self.classify_button = tk.Button(self.left_frame, text="Classify")
        self.classify_button.pack(pady=10)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM items ORDER BY name")
        items = cursor.fetchall()
        conn.close()

        for item in items:
            self.listbox.insert(tk.END, item[0])

    def display_item_info(self, event):
        selected = self.listbox.curselection()
        if not selected:
            return

        item_name = self.listbox.get(selected[0])

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM items WHERE name = ?", (item_name,))
        row = cursor.fetchone()
        col_names = [description[0] for description in cursor.description]
        conn.close()

        self.info_text.config(state='normal')
        self.info_text.delete(1.0, tk.END)
        if row:
            for i, val in enumerate(row):
                if col_names[i] in ("id", "added_on", "photo_path"):
                    continue  # skip photo path listing
                if val:
                    label = col_names[i].replace("_", " ").capitalize()
                    self.info_text.insert(tk.END, f"{label}: {val}\n\n")
        else:
            self.info_text.insert(tk.END, "Item not found.")

        self.info_text.config(state='disabled')

        photo_folder = os.path.join(PHOTO_DIR, item_name.replace(" ", "_"))
        image_paths = []
        if os.path.exists(photo_folder):
            for f in os.listdir(photo_folder):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_paths.append(os.path.join(photo_folder, f))

        self.image_canvas.delete("all")
        self.photo_refs.clear()

        for idx, path in enumerate(image_paths):
            try:
                img = Image.open(path)
                img.thumbnail((300, 300), Image.Resampling.LANCZOS)
                img_tk = ImageTk.PhotoImage(img)
                x = 310 * idx + 10
                self.image_canvas.create_image(
                    x, 10, anchor='nw', image=img_tk)
                self.photo_refs.append(img_tk)
            except Exception as e:
                print(f"Failed to load image: {path}\n{e}")

        self.image_canvas.config(scrollregion=self.image_canvas.bbox("all"))


class ClassifierApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Item Classifier")

        self.image_label = tk.Label(master, text="No image selected")
        self.image_label.pack(pady=10)

        btn_frame = tk.Frame(master)
        btn_frame.pack()

        self.select_button = tk.Button(
            btn_frame, text="Select Image", command=self.select_image)
        self.select_button.grid(row=0, column=0, padx=5)

        self.capture_button = tk.Button(
            btn_frame, text="Capture Image", command=self.capture_image)
        self.capture_button.grid(row=0, column=1, padx=5)

        self.library_button = tk.Button(
            btn_frame, text="View Library", command=self.view_library)
        self.library_button.grid(row=0, column=2, padx=5)

        self.top3_var = tk.BooleanVar()
        self.top3_check = tk.Checkbutton(
            master, text="Show Top 3 Predictions", variable=self.top3_var)
        self.top3_check.pack(pady=(0, 10))

        self.result_text = tk.Text(
            master, height=12, width=80, state='disabled', wrap='word')
        self.result_text.pack(pady=10)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.process_image(file_path)

    def capture_image(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Camera not accessible.")
            return

        cv2.namedWindow("Press SPACE to capture", cv2.WINDOW_NORMAL)
        captured_frame = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Press SPACE to capture", frame)
            key = cv2.waitKey(1)
            if key == 32:
                captured_frame = frame.copy()
                break
            elif key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

        if captured_frame is not None:
            self.process_frame(captured_frame)

    def process_image(self, file_path):
        try:
            img = Image.open(file_path).resize((400, 400))
            tk_img = ImageTk.PhotoImage(img)
            self.image_label.config(image=tk_img, text="")
            self.image_label.image = tk_img

            results = predict_with_description(
                file_path, top_k=3 if self.top3_var.get() else 1)
            self.display_results(results)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def process_frame(self, frame):
        try:
            img = Image.fromarray(cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB)).resize((400, 400))
            tk_img = ImageTk.PhotoImage(img)
            self.image_label.config(image=tk_img, text="")
            self.image_label.image = tk_img

            results = predict_with_description(
                frame, top_k=3 if self.top3_var.get() else 1)
            self.display_results(results)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def display_results(self, results):
        self.result_text.config(state='normal')
        self.result_text.delete(1.0, tk.END)

        for i, (label, confidence, description, fact) in enumerate(results):
            self.result_text.insert(
                tk.END, f"{'🔝 ' if i == 0 else '➡️ '} Prediction {i+1}: {label}\n")
            self.result_text.insert(
                tk.END, f"   🔍 Confidence: {confidence:.2%}\n")
            self.result_text.insert(
                tk.END, f"   📜 Description: {description}\n")
            if fact:
                self.result_text.insert(tk.END, f"   💡 Fun Fact: {fact}\n")
            self.result_text.insert(tk.END, "\n")

        self.result_text.config(state='disabled')

    def view_library(self):
        LibraryWindow(self.master)


if __name__ == "__main__":
    root = tk.Tk()
    app = ClassifierApp(root)
    root.mainloop()
