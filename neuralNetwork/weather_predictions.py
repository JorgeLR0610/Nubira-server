import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import copy
import os

# ------------------ Dataset ------------------
class WeatherDataset(Dataset):
    def __init__(self, csv_file):
        if not os.path.exists(r"Nubira/temp/datos.csv"):
            raise FileNotFoundError(f"No se encontró el archivo: {csv_file}")

        self.data = pd.read_csv(r"Nubira/temp/datos.csv")

        if self.data.empty:
            raise ValueError("El archivo CSV está vacío o mal formateado.")

        # Variables de entrada
        self.features = self.data[["year", "month", "day"]].values
        # Variables objetivo (5 valores)
        self.labels = self.data[["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "PRECSNO", "WS2M_MAX"]].values

        # Escaladores (solo en memoria)
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

        self.features = self.scaler_x.fit_transform(self.features)
        self.labels = self.scaler_y.fit_transform(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


# ------------------ Modelo ------------------
class WeatherPrediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )

    def forward(self, x):
        return self.fc(x)


# ------------------ Entrenamiento y Predicción ------------------
_CACHED = {
    "model": None,
    "best_state": None,
    "scaler_x": None,
    "scaler_y": None,
    "dataset_mtime": None,
}


def predict_by_date(year, month, day, epochs=20, lr=0.03, batch_size=34):
    """
    Entrena el modelo UNA SOLA VEZ, pero guarda el estado (pesos)
    del batch donde el loss fue más bajo en todo el entrenamiento.
    """

    # 📍 Buscar el CSV en la raíz del proyecto
            # Buscar el CSV subiendo un nivel desde la carpeta neuralNetwork
    # Buscar el CSV subiendo un nivel desde la carpeta neuralNetwork


    dataset = WeatherDataset("datos.csv")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 🔁 Reutilizar caché si el dataset no cambió
    reuse_cached = (
        _CACHED["best_state"] is not None and _CACHED["dataset_mtime"] == dataset_mtime
    )

    model = WeatherPrediction()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    best_model_state = None
    best_epoch = 0
    best_batch = 0

    if not reuse_cached:
        for epoch in range(epochs):
            for batch_idx, (x, y) in enumerate(dataloader):
                optimizer.zero_grad()
                y_pred = model(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_epoch = epoch + 1
                    best_batch = batch_idx + 1
                    best_model_state = copy.deepcopy(model.state_dict())
                    print(f"🟢 Nuevo mejor modelo → Época {best_epoch}, Batch {best_batch}, Loss: {best_loss:.6f}")

        _CACHED["best_state"] = best_model_state
        _CACHED["scaler_x"] = dataset.scaler_x
        _CACHED["scaler_y"] = dataset.scaler_y
        _CACHED["dataset_mtime"] = dataset_mtime
    else:
        best_model_state = _CACHED["best_state"]

    model.load_state_dict(best_model_state)
    print(f"✅ Modelo listo (reentrenado={not reuse_cached}). Mejor pérdida: {best_loss:.6f}")

    # Predicción
    x_input = np.array([[year, month, day]])
    scaler_x = _CACHED["scaler_x"] if reuse_cached else dataset.scaler_x
    scaler_y = _CACHED["scaler_y"] if reuse_cached else dataset.scaler_y

    x_scaled = scaler_x.transform(x_input)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    y_pred = model(x_tensor).detach().numpy()
    y_real = scaler_y.inverse_transform(y_pred)

    return {
        "prediccion": y_real[0].tolist(),
        "mejor_loss": best_loss,
        "mejor_epoca": best_epoch,
        "mejor_batch": best_batch
    }
