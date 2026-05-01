import os
import numpy as np
import joblib
import tkinter as tk
from tkinter import messagebox

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_regression_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.joblib")

try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from '{MODEL_PATH}'")
except FileNotFoundError:
    print(f"Could not find model file: {MODEL_PATH}")
    print("Make sure you've trained and saved the model first.")
    exit(1)

try:
    scaler = joblib.load(SCALER_PATH)
    print(f"Scaler loaded successfully from '{SCALER_PATH}'")
except FileNotFoundError:
    print(f"Could not find scaler file: {SCALER_PATH}")
    print("Make sure you've saved the scaler from your notebook.")
    exit(1)

FEATURES = [
    "Global Reactive Power (kW)",
    "Voltage (V)",
    "Global Intensity (A)",
    "Sub Metering 2 (Wh)",
    "Sub Metering 3 (Wh)",
]

def open_prediction_gui():

    root = tk.Tk()
    root.title("Household Energy Consumption Predictor")
    root.geometry("420x450")

    tk.Label(root, text="Enter Sensor Readings", font=("Arial", 14, "bold")).pack(pady=10)

    form_frame = tk.Frame(root)
    form_frame.pack(pady=5)

    entries = {}
    for f in FEATURES:
        row = tk.Frame(form_frame)
        label = tk.Label(row, width=25, text=f + ": ", anchor='w', font=("Arial", 10))
        ent = tk.Entry(row, width=10)
        row.pack(padx=5, pady=4)
        label.pack(side=tk.LEFT)
        ent.pack(side=tk.RIGHT, expand=True)
        entries[f] = ent

    def submit():
        try:
            data = [float(entries[f].get()) for f in FEATURES]
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter numeric values for all fields.")
            return

        arr = np.array(data).reshape(1, -1)
        arr_scaled = scaler.transform(arr)
        pred = model.predict(arr_scaled)[0]

        msg = f"Predicted Global Active Power: {round(pred, 4)} kW"
        messagebox.showinfo("Prediction Result", msg)

    tk.Button(root, text="Submit", command=submit, bg="lightblue", width=15).pack(pady=20)
    root.mainloop()


if __name__ == "__main__":
    open_prediction_gui()