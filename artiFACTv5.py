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

        # Parchment-toned academic explorer color palette with woodgrain
        PARCHMENT_COLORS = {
            'bg_primary': '#D2B48C',      # Tan woodgrain background
            'bg_secondary': '#CD853F',    # Saddle brown for sections
            'accent_gold': '#D4AF37',     # Faint gold for highlights
            'accent_rust': '#B7410E',     # Rust for important elements
            'text_charcoal': '#2F2F2F',   # Charcoal for text
            'text_olive': '#556B2F',      # Olive green for labels
            'button_bg': '#8B4513',       # Saddle brown for buttons
            'button_hover': '#A0522D',    # Rust on hover
            'success_green': '#556B2F',   # Olive green for success
            'error_red': '#8B0000',       # Dark red for errors
            'selection_gold': '#D4AF37',  # Gold for selections
            'border_charcoal': '#2F2F2F', # Charcoal borders
            'text_light': '#4A4A4A',      # Light charcoal for secondary text
            'accent_olive': '#6B8E23',    # Brighter olive for accents
            'title_gold': '#FFD700',      # Bright gold for title
            'title_rust': '#B7410E'       # Rust for title accent
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


class ParchmentButton(tk.Button):
    """Custom parchment-styled button with academic explorer theme"""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(
            bg=PARCHMENT_COLORS['button_bg'],
            fg=PARCHMENT_COLORS['bg_primary'],
            font=('Georgia', 10, 'bold'),
            relief='raised',
            bd=2,
            padx=15,
            pady=8,
            activebackground=PARCHMENT_COLORS['button_hover'],
            activeforeground=PARCHMENT_COLORS['bg_primary'],
            highlightbackground=PARCHMENT_COLORS['border_charcoal'],
            highlightthickness=1
        )
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)

    def on_enter(self, e):
        self.configure(bg=PARCHMENT_COLORS['button_hover'])

    def on_leave(self, e):
        self.configure(bg=PARCHMENT_COLORS['button_bg'])


class ParchmentFrame(tk.Frame):
    """Custom parchment-styled frame with worn texture appearance"""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(
            bg=PARCHMENT_COLORS['bg_secondary'],
            relief='ridge',
            bd=2,
            highlightbackground=PARCHMENT_COLORS['border_charcoal'],
            highlightthickness=1
        )


class ParchmentLabel(tk.Label):
    """Custom parchment-styled label with serif font"""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(
            bg=PARCHMENT_COLORS['bg_secondary'],
            fg=PARCHMENT_COLORS['text_charcoal'],
            font=('Georgia', 11, 'bold'),
            relief='flat',
            bd=1,
            highlightbackground=PARCHMENT_COLORS['border_charcoal'],
            highlightthickness=1
        )


class ParchmentText(tk.Text):
    """Custom parchment-styled text widget"""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(
            bg=PARCHMENT_COLORS['bg_primary'],
            fg=PARCHMENT_COLORS['text_charcoal'],
            font=('Georgia', 10),
            relief='sunken',
            bd=2,
            selectbackground=PARCHMENT_COLORS['selection_gold'],
            selectforeground=PARCHMENT_COLORS['text_charcoal'],
            insertbackground=PARCHMENT_COLORS['accent_rust'],
            highlightbackground=PARCHMENT_COLORS['border_charcoal'],
            highlightthickness=1
        )


class ParchmentListbox(tk.Listbox):
    """Custom parchment-styled listbox"""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(
            bg=PARCHMENT_COLORS['bg_primary'],
            fg=PARCHMENT_COLORS['text_charcoal'],
            font=('Georgia', 10),
            relief='sunken',
            bd=2,
            selectbackground=PARCHMENT_COLORS['selection_gold'],
            selectforeground=PARCHMENT_COLORS['text_charcoal'],
            highlightbackground=PARCHMENT_COLORS['border_charcoal'],
            highlightthickness=1
        )


class ParchmentCanvas(tk.Canvas):
    """Custom parchment-styled canvas"""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(
            bg=PARCHMENT_COLORS['bg_primary'],
            relief='sunken',
            bd=2,
            highlightbackground=PARCHMENT_COLORS['border_charcoal'],
            highlightthickness=1
        )


class LibraryWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("🐚🪲 artiFACTS 🦂🦴")
        self.configure(bg=PARCHMENT_COLORS['bg_primary'])

        # Set to fullscreen - cross-platform compatible
        try:
            # Try Windows zoomed state first
            self.state('zoomed')
        except tk.TclError:
            try:
                # Try Linux zoomed attribute
                self.attributes('-zoomed', True)
            except tk.TclError:
                # Fallback to fullscreen
                self.attributes('-fullscreen', True)

        # Add academic title with styling
        title_container = tk.Frame(self, bg=PARCHMENT_COLORS['bg_secondary'])
        title_container.pack(fill=tk.X, padx=15, pady=10)

        title_label = tk.Label(title_container, 
                              text="🐚🪲 artiFACTS 🦂🦴",
                              font=('Georgia', 28, 'bold'),
                              bg=PARCHMENT_COLORS['bg_secondary'],
                              fg=PARCHMENT_COLORS['title_gold'],
                              relief='raised',
                              bd=2)
        title_label.pack(pady=8)

        main_frame = ParchmentFrame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        self.left_frame = ParchmentFrame(main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Add academic label for listbox
        list_label = ParchmentLabel(self.left_frame, text="📖 ITEM CATALOG",
                                    font=('Georgia', 14, 'bold'))
        list_label.pack(pady=(10, 5))

        self.listbox = ParchmentListbox(self.left_frame, width=45)
        self.listbox.pack(fill=tk.Y, expand=True, padx=10, pady=10)
        self.listbox.bind("<<ListboxSelect>>", self.display_item_info)

        self.right_frame = ParchmentFrame(main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH,
                              expand=True, padx=10, pady=10)

        # Image section with academic styling
        image_label = ParchmentLabel(self.right_frame, text="🖼️ ITEM PHOTOS",
                                     font=('Georgia', 14, 'bold'))
        image_label.pack(pady=(10, 5))

        self.image_frame = ParchmentFrame(self.right_frame)
        self.image_frame.pack(fill='x', pady=(5, 15), padx=10)

        self.image_canvas = ParchmentCanvas(self.image_frame, height=350)
        self.image_canvas.pack(side='top', fill='x',
                               expand=True, padx=10, pady=10)

        self.scroll_x = tk.Scrollbar(
            self.image_frame, orient='horizontal', command=self.image_canvas.xview)
        self.scroll_x.pack(side='bottom', fill='x')

        self.image_canvas.configure(xscrollcommand=self.scroll_x.set)

        self.image_canvas.bind("<ButtonPress-1>", self.start_scroll)
        self.image_canvas.bind("<B1-Motion>", self.do_scroll)

        # Info section with academic styling
        info_label = ParchmentLabel(self.right_frame, text="📋 ITEM DETAILS",
                                     font=('Georgia', 14, 'bold'))
        info_label.pack(pady=(15, 5))

        self.info_text = ParchmentText(self.right_frame, wrap='word')
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

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
                img.thumbnail((320, 320), Image.Resampling.LANCZOS)
                img_tk = ImageTk.PhotoImage(img)
                x = 330 * idx + 15
                self.image_canvas.create_image(
                    x, 15, anchor='nw', image=img_tk)
                self.photo_refs.append(img_tk)
            except Exception as e:
                print(f"Failed to load image: {path}\n{e}")

        self.image_canvas.config(scrollregion=self.image_canvas.bbox("all"))


class ClassifierApp:
    def __init__(self, master):
        self.master = master
        self.master.title("🐚🪲 artiFACTS 🦂🦴")
        self.master.configure(bg=PARCHMENT_COLORS['bg_primary'])

        # Set window to fullscreen - cross-platform compatible
        try:
            # Try Windows zoomed state first
            self.master.state('zoomed')
        except tk.TclError:
            try:
                # Try Linux zoomed attribute
                self.master.attributes('-zoomed', True)
            except tk.TclError:
                # Fallback to fullscreen
                self.master.attributes('-fullscreen', True)

        # Create a canvas for the main window with scrollbars
        self.main_canvas = ParchmentCanvas(self.master)
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
        main_frame = ParchmentFrame(self.main_canvas)
        # Create window with proper width (accounting for scrollbar)
        initial_width = max(1000, self.master.winfo_width() - 50)
        self.main_canvas.create_window(
            (0, 0), window=main_frame, anchor='nw', width=initial_width)

        # Main title with academic styling
        title_frame = ParchmentFrame(main_frame)
        title_frame.pack(fill=tk.X, padx=20, pady=15)

        # Create a custom styled title with gradient effect
        title_container = tk.Frame(title_frame, bg=PARCHMENT_COLORS['bg_secondary'])
        title_container.pack(fill=tk.X, padx=10, pady=10)

        title_label = tk.Label(title_container, 
                              text="🐚🪲 artiFACTS 🦂🦴",
                              font=('Georgia', 36, 'bold'),
                              bg=PARCHMENT_COLORS['bg_secondary'],
                              fg=PARCHMENT_COLORS['title_gold'],
                              relief='raised',
                              bd=3)
        title_label.pack(pady=10)

        subtitle_label = tk.Label(title_container, 
                                 text="Your curiosities library!",
                                 font=('Georgia', 14, 'italic'),
                                 bg=PARCHMENT_COLORS['bg_secondary'],
                                 fg=PARCHMENT_COLORS['title_rust'])
        subtitle_label.pack(pady=(0, 10))

        # Bind mouse wheel scrolling
        self.main_canvas.bind('<MouseWheel>', self.on_mousewheel)
        self.main_canvas.bind('<Button-4>', self.on_mousewheel)
        self.main_canvas.bind('<Button-5>', self.on_mousewheel)

        # Button section with academic styling (moved to second position)
        btn_frame = ParchmentFrame(main_frame)
        btn_frame.pack(fill=tk.X, padx=15, pady=10)

        btn_label = ParchmentLabel(btn_frame, text="🎛️ Research Controls",
                                   font=('Georgia', 14, 'bold'))
        btn_label.pack(pady=(8, 10))

        button_container = tk.Frame(btn_frame, bg=PARCHMENT_COLORS['bg_secondary'])
        button_container.pack(pady=(0, 10))

        self.select_button = ParchmentButton(
            button_container, text="📁 Select Image", command=self.select_image)
        self.select_button.grid(row=0, column=0, padx=12, pady=10)

        self.capture_button = ParchmentButton(
            button_container, text="📷 Capture Image", command=self.capture_image)
        self.capture_button.grid(row=0, column=1, padx=12, pady=10)

        self.library_button = ParchmentButton(
            button_container, text="📚 View Library", command=self.view_library)
        self.library_button.grid(row=0, column=2, padx=12, pady=10)

        # Options section (moved to third position) - reduced padding
        options_frame = ParchmentFrame(main_frame)
        options_frame.pack(fill=tk.X, padx=15, pady=10)

        options_label = ParchmentLabel(options_frame, text="⚙️ OPTIONS",
                                       font=('Georgia', 14, 'bold'))
        options_label.pack(pady=(8, 5))

        self.top3_var = tk.BooleanVar()
        # Create a custom parchment checkbox
        checkbox_frame = tk.Frame(
            options_frame, bg=PARCHMENT_COLORS['bg_secondary'])
        checkbox_frame.pack(pady=8)

        self.top3_check = tk.Checkbutton(
            checkbox_frame, text="Show Top 3 Predictions", variable=self.top3_var,
            bg=PARCHMENT_COLORS['bg_secondary'], fg=PARCHMENT_COLORS['text_charcoal'],
            font=('Georgia', 11, 'bold'), selectcolor=PARCHMENT_COLORS['selection_gold'],
            activebackground=PARCHMENT_COLORS['bg_secondary'],
            activeforeground=PARCHMENT_COLORS['text_charcoal'],
            indicatoron=True,
            relief='raised',
            bd=2
        )
        self.top3_check.pack(pady=8)

        # Image display section (moved to fourth position)
        image_frame = ParchmentFrame(main_frame)
        image_frame.pack(fill=tk.X, padx=15, pady=10)

        image_label = ParchmentLabel(image_frame, text="🖼️ SELECTED IMAGE",
                                     font=('Georgia', 14, 'bold'))
        image_label.pack(pady=(8, 5))

        self.image_label = ParchmentLabel(image_frame, text="No image selected",
                                          font=('Georgia', 12))
        self.image_label.pack(pady=10)

        # Results section (stays in place) - reduced height
        results_frame = ParchmentFrame(main_frame)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        results_label = ParchmentLabel(
            results_frame, text="🔍 CLASSIFICATION RESULTS",
             font=('Georgia', 14, 'bold'))
        results_label.pack(pady=(8, 5))

        self.result_text = ParchmentText(
            results_frame, height=12, width=90, state='disabled', wrap='word')
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
            img = Image.open(file_path).resize((350, 350))
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
                frame, cv2.COLOR_BGR2RGB)).resize((350, 350))
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
