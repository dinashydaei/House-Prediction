import os
import sys
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk
import joblib
import numpy as np
import pandas as pd

from PIL import Image, ImageDraw, ImageFilter, ImageTk
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# CONFIGURATION
# =========================================================

APP_NAME = "House Price Prediction"
DATASET_NAME = "HouseNew(1).csv"
MODEL_NAME = "house_price_model.joblib"

BG = "#F3F4F6"
PANEL = "#FFFFFF"
CARD = "#F8F9FA"
INPUT_BG = "#FFFFFF"
BLACK = "#151619"
BLACK_SOFT = "#303238"
MUTED = "#777B83"
BORDER = "#E0E2E6"
WHITE = "#FFFFFF"

FEATURES = [
    "Elevator",
    "Floor",
    "Area",
    "Parking",
    "Room",
    "Warehouse",
    "YearOfConstruction",
    "Address"
]

NUMERIC = [
    "Floor",
    "Area",
    "Room",
    "YearOfConstruction"
]

CATEGORICAL = [
    "Elevator",
    "Parking",
    "Warehouse",
    "Address"
]


# =========================================================
# FILE PATHS
# =========================================================

def resource_path(filename):
    """Find a bundled resource or a file beside the Python script."""

    candidates = []

    if getattr(sys, "_MEIPASS", None):
        candidates.append(Path(sys._MEIPASS) / filename)

    if getattr(sys, "frozen", False):
        candidates.append(
            Path(sys.executable).resolve().parent / filename
        )

    candidates.append(Path(__file__).resolve().parent / filename)

    for path in candidates:
        if path.exists():
            return path

    return candidates[-1]


def writable_path(filename):
    """Save the trained model beside the executable or source file."""

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / filename

    return Path(__file__).resolve().parent / filename


# =========================================================
# MINIMAL BACKGROUND
# =========================================================

def create_background_image(width=850, height=760):
    """Create a clean white background with subtle gray accents."""

    image = Image.new("RGB", (width, height), BG)
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    draw.ellipse(
        (width - 230, -160, width + 100, 170),
        fill=(210, 213, 218, 105)
    )

    draw.ellipse(
        (-160, height - 210, 170, height + 100),
        fill=(222, 224, 228, 95)
    )

    overlay = overlay.filter(ImageFilter.GaussianBlur(55))
    image = Image.alpha_composite(image.convert("RGBA"), overlay)

    return image.convert("RGB")


# =========================================================
# MACHINE LEARNING
# =========================================================

class HousePriceModel:

    def __init__(self):
        self.model_path = writable_path(MODEL_NAME)
        self.model = None
        self.columns = None

        if self.model_path.exists():
            try:
                self.load_model()
                print("Joblib model loaded successfully.")
                return
            except Exception as error:
                print("Saved model could not be loaded:", error)

        self.train_model()
        self.save_model()

    @staticmethod
    def normalize_boolean(value):
        if isinstance(value, (bool, np.bool_)):
            return "Yes" if value else "No"

        text = str(value).strip().lower()

        if text in {"yes", "true", "1", "بله"}:
            return "Yes"

        return "No"

    def load_dataset(self):
        path = resource_path(DATASET_NAME)

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found:\n{path}\n\n"
                f"Place {DATASET_NAME} beside the Python file."
            )

        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="cp1256")

        df.columns = df.columns.astype(str).str.strip()

        required = FEATURES + ["Price"]

        missing = [col for col in required if col not in df.columns]

        if missing:
            raise ValueError(
                "The following columns are missing from the dataset:\n"
                + ", ".join(missing)
            )

        df = df[required].copy()

        for col in NUMERIC + ["Price"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["Price", "Area"])
        df = df[(df["Price"] > 0) & (df["Area"] > 0)]

        for col in NUMERIC:
            median = df[col].median()

            if pd.isna(median):
                median = 0

            df[col] = df[col].fillna(median)

        for col in ["Elevator", "Parking", "Warehouse"]:
            df[col] = df[col].map(self.normalize_boolean)

        df["Address"] = (
            df["Address"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )

        if len(df) < 20:
            raise ValueError(
                "The dataset contains fewer than 20 valid records."
            )

        return df

    def encode_features(self, df, fit=False):
        encoded = pd.get_dummies(
            df[FEATURES],
            columns=CATEGORICAL,
            dtype=float
        )

        if fit:
            self.columns = encoded.columns.tolist()
        else:
            encoded = encoded.reindex(
                columns=self.columns,
                fill_value=0.0
            )

        return encoded

    def train_model(self):
        df = self.load_dataset()

        X = self.encode_features(df, fit=True)
        y = np.log1p(df["Price"].astype(float))

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )

        # Increasing area should not decrease predicted price.
        # A newer construction year should not decrease price.
        constraints = {
            column: 0 for column in X.columns
        }

        constraints["Area"] = 1
        constraints["YearOfConstruction"] = 1

        self.model = HistGradientBoostingRegressor(
            max_iter=180,
            learning_rate=0.08,
            max_leaf_nodes=25,
            l2_regularization=1.0,
            monotonic_cst=constraints,
            random_state=42
        )

        self.model.fit(X_train, y_train)

        predictions = np.maximum(
            0,
            np.expm1(self.model.predict(X_test))
        )

        actual = np.expm1(y_test)

        print("\nMODEL EVALUATION")
        print("-" * 35)
        print(
            "MAE:",
            round(mean_absolute_error(actual, predictions), 2)
        )
        print(
            "RMSE:",
            round(np.sqrt(mean_squared_error(actual, predictions)), 2)
        )
        print("R2 Score:", round(r2_score(actual, predictions), 4))
        print("Valid records:", len(df))

    def save_model(self):
        artifact = {
            "version": 1,
            "model": self.model,
            "columns": self.columns
        }

        joblib.dump(
            artifact,
            self.model_path,
            compress=3
        )

        print("Model saved to:", self.model_path)

    def load_model(self):
        artifact = joblib.load(self.model_path)

        if artifact.get("version") != 1:
            raise ValueError("Unsupported model version.")

        self.model = artifact["model"]
        self.columns = artifact["columns"]

        if self.model is None or not self.columns:
            raise ValueError("The saved model is incomplete.")

    def predict(self, values):
        row = pd.DataFrame([values], columns=FEATURES)

        for col in ["Elevator", "Parking", "Warehouse"]:
            row[col] = row[col].map(self.normalize_boolean)

        row["Address"] = (
            row["Address"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )

        X = self.encode_features(row, fit=False)

        predicted_log_price = float(self.model.predict(X)[0])

        return max(0.0, float(np.expm1(predicted_log_price)))


# =========================================================
# USER INTERFACE
# =========================================================

class HousePriceApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("dark-blue")

        self.title(APP_NAME)
        self.geometry("300x660")
        self.minsize(550, 500)
        self.configure(fg_color=BG)

        self.model = None
        self.entries = {}
        self.radio_vars = {}

        self.background_photo = None
        self.background_label = None

        self.create_background()
        self.create_interface()

        try:
            self.model = HousePriceModel()
        except Exception as error:
            messagebox.showerror(
                "Model Error",
                str(error),
                parent=self
            )

    def create_background(self):
        background = create_background_image(850, 760)

        self.background_photo = ImageTk.PhotoImage(background)

        self.background_label = tk.Label(
            self,
            image=self.background_photo,
            borderwidth=0,
            highlightthickness=0
        )

        self.background_label.place(
            x=0,
            y=0,
            relwidth=1,
            relheight=1
        )

        self.background_label.lower()

    def create_interface(self):
        # Main application panel.
        self.main = ctk.CTkFrame(
            self,
            fg_color=PANEL,
            corner_radius=20,
            border_width=1,
            border_color=BORDER
        )

        self.main.pack(
            fill="both",
            expand=True,
            padx=12,
            pady=12
        )

        # Header icon.
        ctk.CTkLabel(
            self.main,
            text="⌂",
            font=("Segoe UI Symbol", 38),
            text_color=BLACK
        ).pack(pady=(5, 0))

        ctk.CTkLabel(
            self.main,
            text="PROPERTY ANALYTICS",
            font=("Arial", 9, "bold"),
            text_color=MUTED
        ).pack(pady=(1, 2))

        ctk.CTkLabel(
            self.main,
            text="House Price Prediction",
            font=("Georgia", 27, "bold"),
            text_color=BLACK
        ).pack(pady=(0, 3))

        ctk.CTkLabel(
            self.main,
            text="INTELLIGENT VALUATION  •  REAL ESTATE",
            font=("Arial", 9),
            text_color=MUTED
        ).pack(pady=(0, 12))

        # Form container.
        self.form = ctk.CTkFrame(
            self.main,
            fg_color=CARD,
            corner_radius=16,
            border_width=1,
            border_color=BORDER
        )

        self.form.pack(
            fill="both",
            expand=True,
            padx=16,
            pady=(0, 8)
        )

        self.form.grid_columnconfigure(0, weight=1)
        self.form.grid_columnconfigure(1, weight=1)

        self.add_radio_row(
            "♙  Elevator",
            "Elevator",
            0,
            default="No"
        )

        self.add_entry_row(
            "▤   Floor",
            "Floor",
            1,
            "Enter floor number"
        )

        self.add_entry_row(
            "▱   Area (m²)",
            "Area",
            2,
            "Enter property area"
        )

        self.add_radio_row(
            "▣   Parking",
            "Parking",
            3,
            default="Yes"
        )

        self.add_entry_row(
            "▥   Rooms",
            "Room",
            4,
            "Number of rooms"
        )

        self.add_radio_row(
            "⌂   Warehouse",
            "Warehouse",
            5,
            default="Yes"
        )

        self.add_entry_row(
            "▦   Construction Year",
            "YearOfConstruction",
            6,
            "e.g. 1400"
        )

        self.add_entry_row(
            "⌖   Address",
            "Address",
            7,
            "Address..."
        )

        # Buttons.
        buttons = ctk.CTkFrame(
            self.form,
            fg_color="transparent"
        )

        buttons.grid(
            row=8,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=18,
            pady=(12, 10)
        )

        buttons.grid_columnconfigure(0, weight=1)
        buttons.grid_columnconfigure(1, weight=1)

        ctk.CTkButton(
            buttons,
            text="↺   Reset",
            height=42,
            corner_radius=11,
            fg_color=WHITE,
            hover_color="#E8E9EC",
            border_width=1,
            border_color=BORDER,
            text_color=BLACK,
            font=("Arial", 13),
            command=self.reset_form
        ).grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 8)
        )

        ctk.CTkButton(
            buttons,
            text="⌂   Predict Price",
            height=42,
            corner_radius=11,
            fg_color=BLACK,
            hover_color=BLACK_SOFT,
            text_color=WHITE,
            font=("Arial", 13, "bold"),
            command=self.predict_price
        ).grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(8, 0)
        )

        # Prediction result card.
        self.result_card = ctk.CTkFrame(
            self.form,
            fg_color=WHITE,
            corner_radius=13,
            border_width=1,
            border_color=BORDER
        )

        self.result_card.grid(
            row=9,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=18,
            pady=(0, 12)
        )

        self.result_card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            self.result_card,
            text="⌂",
            font=("Segoe UI Symbol", 30),
            text_color=BLACK
        ).grid(
            row=0,
            column=0,
            padx=(14, 10),
            pady=10
        )

        result_text = ctk.CTkFrame(
            self.result_card,
            fg_color="transparent"
        )

        result_text.grid(
            row=0,
            column=1,
            sticky="w",
            pady=8
        )

        ctk.CTkLabel(
            result_text,
            text="ESTIMATED HOUSE PRICE",
            font=("Arial", 9, "bold"),
            text_color=MUTED
        ).pack(anchor="w")

        self.price_label = ctk.CTkLabel(
            result_text,
            text="Enter details to predict",
            font=("Arial", 17, "bold"),
            text_color=BLACK
        )

        self.price_label.pack(
            anchor="w",
            pady=(3, 0)
        )

        ctk.CTkLabel(
            self.result_card,
            text="Toman :)",
            font=("Arial", 10),
            text_color=MUTED
        ).grid(
            row=0,
            column=2,
            padx=14
        )

        # Footer.


    def add_entry_row(self, label, key, row, placeholder):
        container = ctk.CTkFrame(
            self.form,
            fg_color="transparent"
        )

        container.grid(
            row=row,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=18,
            pady=4
        )

        container.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            container,
            text=label,
            font=("Arial", 12),
            text_color=BLACK,
            anchor="w",
            width=205
        ).grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 10)
        )

        entry = ctk.CTkEntry(
            container,
            height=36,
            corner_radius=9,
            border_width=1,
            border_color=BORDER,
            fg_color=INPUT_BG,
            text_color=BLACK,
            placeholder_text=placeholder,
            placeholder_text_color="#92969D",
            font=("Arial", 12)
        )

        entry.grid(
            row=0,
            column=1,
            sticky="ew"
        )

        self.entries[key] = entry

    def add_radio_row(self, label, key, row, default="Yes"):
        container = ctk.CTkFrame(
            self.form,
            fg_color="transparent"
        )

        container.grid(
            row=row,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=18,
            pady=4
        )

        ctk.CTkLabel(
            container,
            text=label,
            font=("Arial", 12),
            text_color=BLACK,
            width=205,
            anchor="w"
        ).pack(
            side="left",
            padx=(0, 10)
        )

        variable = tk.StringVar(value=default)
        self.radio_vars[key] = variable

        for value in ("Yes", "No"):
            ctk.CTkRadioButton(
                container,
                text=value,
                variable=variable,
                value=value,
                radiobutton_width=17,
                radiobutton_height=17,
                border_width_checked=5,
                fg_color=BLACK,
                hover_color=BLACK_SOFT,
                border_color="#8C9097",
                text_color=BLACK,
                font=("Arial", 11)
            ).pack(
                side="left",
                padx=(8, 18)
            )

    def reset_form(self):
        for entry in self.entries.values():
            entry.delete(0, "end")

        self.radio_vars["Elevator"].set("No")
        self.radio_vars["Parking"].set("Yes")
        self.radio_vars["Warehouse"].set("Yes")

        self.price_label.configure(
            text="Enter details to predict",
            text_color=BLACK
        )

    def predict_price(self):
        if self.model is None:
            messagebox.showerror(
                "Model Unavailable",
                "The model could not be loaded or trained.",
                parent=self
            )
            return

        try:
            floor = float(self.entries["Floor"].get())
            area = float(self.entries["Area"].get())
            rooms = float(self.entries["Room"].get())
            year = float(
                self.entries["YearOfConstruction"].get()
            )

            address = self.entries["Address"].get().strip()

            if floor < 0:
                raise ValueError("Floor cannot be negative.")

            if area <= 0:
                raise ValueError("Area must be greater than zero.")

            if rooms < 0:
                raise ValueError("Room count cannot be negative.")

            if not 1300 <= year <= 1500:
                raise ValueError(
                    "Enter a valid construction year, e.g. 1398."
                )

            if not address:
                address = "Unknown"

            house = {
                "Elevator": self.radio_vars["Elevator"].get(),
                "Floor": floor,
                "Area": area,
                "Parking": self.radio_vars["Parking"].get(),
                "Room": rooms,
                "Warehouse": self.radio_vars["Warehouse"].get(),
                "YearOfConstruction": year,
                "Address": address
            }

            price = self.model.predict(house)

            self.price_label.configure(
                text=f"{price:,.0f}",
                text_color=BLACK
            )

        except ValueError as error:
            messagebox.showwarning(
                "Invalid Information",
                str(error),
                parent=self
            )

        except Exception as error:
            messagebox.showerror(
                "Prediction Error",
                str(error),
                parent=self
            )


# =========================================================
# APPLICATION ENTRY POINT
# =========================================================

if __name__ == "__main__":
    app = HousePriceApp()
    app.mainloop()