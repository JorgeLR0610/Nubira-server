import torch
from torch import nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import copy
from .database_operations import read_data

# ------------------ Dataset SQL ------------------
class WeatherDatasetSQL(Dataset):
    def __init__(self):
        df = read_data()
        if df.empty:
            raise ValueError("No hay datos en la base de datos.")

        self.features = df[["year", "month", "day"]].values
        self.labels = df[["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "PRECSNO", "WS2M_MAX"]].values
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.features = self.scaler_x.fit_transform(self.features)
        self.labels = self.scaler_y.fit_transform(self.labels)
        self.dataset_mtime = 0  # opcional, para control de caché

    def __len__(self):
        return len(self.features)

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

# ------------------ Caché ------------------
_CACHED = {
    "best_state": None,
    "scaler_x": None,
    "scaler_y": None,
    "dataset_mtime": None
}

# ------------------ Predicción ------------------
def predict_by_date(year, month, day, epochs=80, lr=0.03, batch_size=34):
    dataset = WeatherDatasetSQL()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    reuse_cached = _CACHED["best_state"] is not None and _CACHED["dataset_mtime"] == dataset.dataset_mtime

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
        _CACHED["dataset_mtime"] = dataset.dataset_mtime
    else:
        best_model_state = _CACHED["best_state"]

    model.load_state_dict(best_model_state)

    x_input = np.array([[year, month, day]])
    scaler_x = _CACHED["scaler_x"]
    scaler_y = _CACHED["scaler_y"]
    x_scaled = scaler_x.transform(x_input)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    y_pred = model(x_tensor).detach().numpy()
    y_real = scaler_y.inverse_transform(y_pred)

    print("x_input:", x_input)
    print("x_scaled:", x_scaled)
    print("y_pred:", y_pred)
    print("y_real:", y_real)


    return {
        "prediccion": y_real[0].tolist(),
        "mejor_loss": best_loss,
        "mejor_epoca": best_epoch,
        "mejor_batch": best_batch
    }

