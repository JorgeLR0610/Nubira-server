import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
from .database_operations import read_data
from sklearn.preprocessing import StandardScaler

class WeatherDatasetSQL(torch.utils.data.Dataset):
    def __init__(self):
        df = read_data()
        if df.empty:
            raise ValueError("No hay datos en la base de datos.")

        # Features y labels
        self.features = df[["year", "month", "day"]].values
        self.labels = df[["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "PRECSNO", "WS2M_MAX"]].values

        # Escalado
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.features = self.scaler_x.fit_transform(self.features)
        self.labels = self.scaler_y.fit_transform(self.labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


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


def predict_by_date(year, month, day, epochs=20, lr=0.03, batch_size=34):
    dataset = WeatherDatasetSQL()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = WeatherPrediction()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    best_model_state = None
    best_epoch = 0
    best_batch = 0

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

    model.load_state_dict(best_model_state)

    # Predicción
    x_input = np.array([[year, month, day]])
    x_scaled = dataset.scaler_x.transform(x_input)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    y_pred = model(x_tensor).detach().numpy()

    # Invertir escalado de forma segura
    try:
        y_real = dataset.scaler_y.inverse_transform(y_pred)
    except Exception as e:
        print("⚠️ Error al hacer inverse_transform:", e)
        y_real = y_pred  # fallback por seguridad

    # Limpiar valores que rompen JSON
    y_real = np.nan_to_num(y_real, nan=0.0, posinf=1e6, neginf=-1e6)

    # Convertir a float estándar
    y_real = y_real.astype(float)

    return {
        "prediccion": y_real[0].tolist(),
        "mejor_loss": float(best_loss),
        "mejor_epoca": int(best_epoch),
        "mejor_batch": int(best_batch)
    }
