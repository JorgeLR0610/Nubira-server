import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
import joblib

PROJECT_DIR = Path(__file__).resolve().parent.parent
TEMP_DIR = PROJECT_DIR / "temp"
DEFAULT_CSV = TEMP_DIR / "datos.csv"


# Dataset
class WeatherDataset(Dataset):
    def __init__(self, csv_file=None):
        target_csv = Path(csv_file) if csv_file else DEFAULT_CSV
        if not target_csv.exists():
            raise FileNotFoundError(f"No se encontró el archivo de datos en {target_csv}")

        self.data = pd.read_csv(target_csv)

        # Features: year, monthOfYear/month, day (día del mes)
        month_col = "monthOfYear" if "monthOfYear" in self.data.columns else "month"
        self.features = self.data[["year", month_col, "day"]].values
        # Labels: T2M_MAX, T2M_MIN, PRECTOTCORR
        self.labels = self.data[["T2M_MAX", "T2M_MIN", "PRECTOTCORR"]].values

        # Escaladores
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


# Modelo
class WeatherPrediction(nn.Module):
    def __init__(self):
        super(WeatherPrediction, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 64),   # 3 features: year, month, day
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 3)   # 3 salidas: T2M_MAX, T2M_MIN, PRECTOTCORR
        )

    def forward(self, x):
        return self.fc(x)


def train_save(dataset):
    # Dataset y DataLoader
    dataloader = DataLoader(dataset, batch_size=27, shuffle=True)

    # Modelo, criterio y optimizador
    model = WeatherPrediction()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Entrenamiento
    for epoch in range(100):
        loss = torch.tensor(0.0)
        for x, y in dataloader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    save_dir = PROJECT_DIR / "workspace"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "weather_model.pth")
    joblib.dump(dataset.scaler_x, save_dir / "scaler_x.save")
    joblib.dump(dataset.scaler_y, save_dir / "scaler_y.save")
