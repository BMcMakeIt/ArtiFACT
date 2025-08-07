import tkinter as tk
from tkinter import filedialog, messagebox, ttk
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

# Retro color palette - Vintage Computer Green Theme
RETRO_COLORS = {
    'bg_primary': '#0A0A0A',      # Deep black (like old CRT screens)
    'bg_secondary': '#1A2F1A',    # Dark green-gray
    'accent_green': '#00FF00',    # Bright green (classic terminal green)
    'accent_cyan': '#00FFFF',     # Cyan for highlights
    'text_light': '#C0C0C0',      # Light gray (like old monitors)
    'text_dark': '#000000',       # Pure black
    'button_bg': '#2F4F2F',       # Dark green
    'button_hover': '#4F6F4F',    # Lighter green on hover
    'success_green': '#00FF00',   # Bright green for success
    'error_red': '#FF0000',       # Bright red for errors
    'selection_green': '#00FF00',  # Bright green for selections
    'border_green': '#00FF00'     # Green borders
}

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


class RetroButton(tk.Button):
    """Custom retro-styled button"""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(
            bg=RETRO_COLORS['button_bg'],
            fg=RETRO_COLORS['accent_green'],
            font=('Courier', 10, 'bold'),
            relief='raised',
            bd=3,
            padx=15,
            pady=8,
            activebackground=RETRO_COLORS['button_hover'],
            activeforeground=RETRO_COLORS['accent_green'],
            highlightbackground=RETRO_COLORS['border_green'],
            highlightthickness=2
        )
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)

    def on_enter(self, e):
        self.configure(bg=RETRO_COLORS['button_hover'])

    def on_leave(self, e):
        self.configure(bg=RETRO_COLORS['button_bg'])


class RetroFrame(tk.Frame):
    """Custom retro-styled frame with border"""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(
            bg=RETRO_COLORS['bg_secondary'],
            relief='ridge',
            bd=4,
            highlightbackground=RETRO_COLORS['border_green'],
            highlightthickness=2
        )


class RetroLabel(tk.Label):
    """Custom retro-styled label"""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(
            bg=RETRO_COLORS['bg_secondary'],
            fg=RETRO_COLORS['accent_green'],
            font=('Courier', 11, 'bold'),
            relief='sunken',
            bd=2,
            highlightbackground=RETRO_COLORS['border_green'],
            highlightthickness=1
        )


class RetroText(tk.Text):
    """Custom retro-styled text widget"""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(
            bg=RETRO_COLORS['text_dark'],
            fg=RETRO_COLORS['accent_green'],
            font=('Courier', 10),
            relief='sunken',
            bd=3,
            selectbackground=RETRO_COLORS['selection_green'],
            selectforeground=RETRO_COLORS['text_dark'],
            insertbackground=RETRO_COLORS['accent_green'],
            highlightbackground=RETRO_COLORS['border_green'],
            highlightthickness=2
        )


class RetroListbox(tk.Listbox):
    """Custom retro-styled listbox"""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(
            bg=RETRO_COLORS['text_dark'],
            fg=RETRO_COLORS['accent_green'],
            font=('Courier', 10),
            relief='sunken',
            bd=3,
            selectbackground=RETRO_COLORS['selection_green'],
            selectforeground=RETRO_COLORS['text_dark'],
            highlightbackground=RETRO_COLORS['border_green'],
            highlightthickness=2
        )


class RetroCanvas(tk.Canvas):
    """Custom retro-styled canvas"""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(
            bg=RETRO_COLORS['text_dark'],
            relief='sunken',
            bd=3,
            highlightbackground=RETRO_COLORS['border_green'],
            highlightthickness=2
        )


class LibraryWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("🐚🪲 artiFACTS 🦂🦴")
        self.configure(bg=RETRO_COLORS['bg_primary'])

        self.geometry("1200x700")

        # Add retro title
        title_label = RetroLabel(self, text="🐚🪲 artiFACTS 🦂🦴",
                                 font=('Courier', 22, 'bold'))
        title_label.pack(pady=10)

        main_frame = RetroFrame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.left_frame = RetroFrame(main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Add retro label for listbox
        list_label = RetroLabel(self.left_frame, text="📖 ITEM CATALOG")
        list_label.pack(pady=(5, 0))

        self.listbox = RetroListbox(self.left_frame, width=40)
        self.listbox.pack(fill=tk.Y, expand=True, padx=5, pady=5)
        self.listbox.bind("<<ListboxSelect>>", self.display_item_info)

        self.right_frame = RetroFrame(main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH,
                              expand=True, padx=5, pady=5)

        # Image section with retro styling
        image_label = RetroLabel(self.right_frame, text="🖼️ ITEM PHOTOS")
        image_label.pack(pady=(5, 0))

        self.image_frame = RetroFrame(self.right_frame)
        self.image_frame.pack(fill='x', pady=(5, 10), padx=5)

        self.image_canvas = RetroCanvas(self.image_frame, height=320)
        self.image_canvas.pack(side='top', fill='x',
                               expand=True, padx=5, pady=5)

        self.scroll_x = tk.Scrollbar(
            self.image_frame, orient='horizontal', command=self.image_canvas.xview)
        self.scroll_x.pack(side='bottom', fill='x')

        self.image_canvas.configure(xscrollcommand=self.scroll_x.set)

        self.image_canvas.bind("<ButtonPress-1>", self.start_scroll)
        self.image_canvas.bind("<B1-Motion>", self.do_scroll)

        # Info section with retro styling
        info_label = RetroLabel(self.right_frame, text="📋 ITEM DETAILS")
        info_label.pack(pady=(10, 0))

        self.info_text = RetroText(self.right_frame, wrap='word')
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.photo_refs = []

        self.load_items()

    def start_scroll(self, event):
        self.image_canvas.scan_mark(event.x, event.y)

    def do_scroll(self, event):
        self.image_canvas.scan_dragto(event.x, event.y, gain=1)

    def load_items(self):
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
                    self.info_text.insert(tk.END, f"🎯 {label}: {val}\n\n")
        else:
            self.info_text.insert(tk.END, "❌ Item not found.")
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
        self.master.title("🐚🪲 artiFACTS 🦂🦴")
        self.master.configure(bg=RETRO_COLORS['bg_primary'])

        # Set window icon and styling
        self.master.geometry("900x700")

        # Create a canvas for the main window with scrollbars
        self.main_canvas = RetroCanvas(self.master)
        self.main_canvas.pack(side='left', fill='both', expand=True)

        # Add scrollbars to main window
        self.main_scroll_x = tk.Scrollbar(
            self.master, orient='horizontal', command=self.main_canvas.xview)
        self.main_scroll_x.pack(side='bottom', fill='x')

        self.main_scroll_y = tk.Scrollbar(
            self.master, orient='vertical', command=self.main_canvas.yview)
        self.main_scroll_y.pack(side='right', fill='y')

        self.main_canvas.configure(
            xscrollcommand=self.main_scroll_x.set,
            yscrollcommand=self.main_scroll_y.set
        )

        # Main content frame inside canvas
        main_frame = RetroFrame(self.main_canvas)
        # Create window with proper width (accounting for scrollbar)
        initial_width = max(800, self.master.winfo_width() - 50)
        self.main_canvas.create_window(
            (0, 0), window=main_frame, anchor='nw', width=initial_width)

        # Main title with retro styling (now inside the canvas)
        title_frame = RetroFrame(main_frame)
        title_frame.pack(fill=tk.X, padx=20, pady=20)

        title_label = RetroLabel(title_frame, text="🐚🪲 artiFACTS 🦂🦴",
                                 font=('Courier', 22, 'bold'))
        title_label.pack(pady=15)

        subtitle_label = RetroLabel(title_frame, text="Your curiosities library!",
                                    font=('Courier', 12))
        subtitle_label.pack(pady=(0, 15))

        # Bind mouse wheel scrolling
        self.main_canvas.bind('<MouseWheel>', self.on_mousewheel)
        self.main_canvas.bind('<Button-4>', self.on_mousewheel)
        self.main_canvas.bind('<Button-5>', self.on_mousewheel)

        # Image display section
        image_frame = RetroFrame(main_frame)
        image_frame.pack(fill=tk.X, padx=10, pady=10)

        image_label = RetroLabel(image_frame, text="🖼️ SELECTED IMAGE")
        image_label.pack(pady=(5, 0))

        self.image_label = RetroLabel(image_frame, text="No image selected",
                                      font=('Courier', 12))
        self.image_label.pack(pady=10)

        # Button section with retro styling
        btn_frame = RetroFrame(main_frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        btn_label = RetroLabel(btn_frame, text="🎛️ CONTROL PANEL")
        btn_label.pack(pady=(5, 10))

        button_container = tk.Frame(btn_frame, bg=RETRO_COLORS['bg_secondary'])
        button_container.pack(pady=(0, 10))

        self.select_button = RetroButton(
            button_container, text="📁 Select Image", command=self.select_image)
        self.select_button.grid(row=0, column=0, padx=10, pady=10)

        self.capture_button = RetroButton(
            button_container, text="📷 Capture Image", command=self.capture_image)
        self.capture_button.grid(row=0, column=1, padx=10, pady=10)

        self.library_button = RetroButton(
            button_container, text="📚 View Library", command=self.view_library)
        self.library_button.grid(row=0, column=2, padx=10, pady=10)

        # Options section
        options_frame = RetroFrame(main_frame)
        options_frame.pack(fill=tk.X, padx=10, pady=10)

        options_label = RetroLabel(options_frame, text="⚙️ OPTIONS")
        options_label.pack(pady=(5, 0))

        self.top3_var = tk.BooleanVar()
        # Create a custom retro checkbox with better visual indication
        checkbox_frame = tk.Frame(
            options_frame, bg=RETRO_COLORS['bg_secondary'])
        checkbox_frame.pack(pady=10)

        self.top3_check = tk.Checkbutton(
            checkbox_frame, text="Show Top 3 Predictions", variable=self.top3_var,
            bg=RETRO_COLORS['bg_secondary'], fg=RETRO_COLORS['accent_green'],
            font=('Courier', 10, 'bold'), selectcolor=RETRO_COLORS['selection_green'],
            activebackground=RETRO_COLORS['bg_secondary'],
            activeforeground=RETRO_COLORS['accent_green'],
            indicatoron=True,  # Show the checkbox indicator
            relief='raised',    # Raised appearance
            bd=2               # Border width for better visibility
        )
        self.top3_check.pack(pady=10)

        # Results section
        results_frame = RetroFrame(main_frame)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        results_label = RetroLabel(
            results_frame, text="🔍 CLASSIFICATION RESULTS")
        results_label.pack(pady=(5, 0))

        self.result_text = RetroText(
            results_frame, height=12, width=80, state='disabled', wrap='word')
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Bind main frame resize to update scroll region
        main_frame.bind('<Configure>',
                        lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all")))

        # Update canvas window width when main window resizes
        self.master.bind('<Configure>', self.on_window_resize)

    def on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        if event.num == 4 or event.delta > 0:
            self.main_canvas.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:
            self.main_canvas.yview_scroll(1, "units")

    def on_window_resize(self, event):
        """Update canvas window width when main window resizes"""
        if hasattr(self, 'main_canvas') and event.widget == self.master:
            # Update the canvas window to match the new window width
            canvas_width = event.width - 50  # Account for scrollbar width
            # Update the first (and only) window item
            self.main_canvas.itemconfig(1, width=canvas_width)

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
            img = Image.open(file_path).resize(
                (320, 320))  # Reduced by 20% from 400x400
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
                frame, cv2.COLOR_BGR2RGB)).resize((320, 320))  # Reduced by 20% from 400x400
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
            if i == 0:
                self.result_text.insert(tk.END, f"🏆 TOP PREDICTION: {label}\n")
            else:
                self.result_text.insert(
                    tk.END, f"🥈 PREDICTION {i+1}: {label}\n")

            self.result_text.insert(
                tk.END, f"   🎯 Confidence: {confidence:.2%}\n")
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
