import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import brier_score_loss, roc_auc_score
from tqdm import tqdm

from config import (
    BEST_MODEL_PATH,
    TRAIN_END_YEAR,
    VAL_END_YEAR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    SAMPLES_PER_EPOCH,
    VAL_SAMPLES,
    TEST_SAMPLES,
    NUM_WORKERS,
    SEED,
)
from dataset import load_daily_dataset_and_context, DailyRiskDataset
from model import WeatherRiskProbNet

TARGET_NAMES = ["Lluvia", "Viento Fuerte", "Temperaturas Adversas"]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_probabilistic(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for x_patch, x_meta, y in loader:
            x_patch = x_patch.to(device, non_blocking=True)
            x_meta = x_meta.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x_patch, x_meta)
            loss = criterion(logits, y)
            batch_n = y.size(0)
            total_loss += loss.item() * batch_n
            total_count += batch_n
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(y.cpu().numpy())

    probs_np = np.concatenate(all_probs, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)
    metrics = {"loss": total_loss / total_count, "per_target": {}}
    for idx, name in enumerate(TARGET_NAMES):
        y_true = targets_np[:, idx]
        y_prob = probs_np[:, idx]
        brier = float(brier_score_loss(y_true, y_prob))
        auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
        metrics["per_target"][name] = {
            "brier": brier,
            "roc_auc": auc,
            "mean_pred_pct": float(y_prob.mean() * 100.0),
            "real_freq_pct": float(y_true.mean() * 100.0),
        }
    return metrics, probs_np, targets_np


def print_calibration_report(probs_np: np.ndarray, targets_np: np.ndarray) -> None:
    """Verifica que los porcentajes predichos coincidan con la frecuencia real observada."""
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    print("\n--- Verificación de Calibración en Años de Prueba (2020-2022) ---")
    for idx, name in enumerate(TARGET_NAMES):
        print(f"\nRiesgo: {name}")
        y_prob = probs_np[:, idx]
        y_true = targets_np[:, idx]
        for low, high in bins:
            mask = (y_prob >= low) & (y_prob < high)
            if mask.sum() == 0:
                continue
            pred_avg = y_prob[mask].mean() * 100.0
            real_avg = y_true[mask].mean() * 100.0
            print(
                f"  Rango {int(low*100):2d}%-{int(min(high, 1.0)*100):3d}% | "
                f"Predicho: {pred_avg:5.1f}% vs Real Observado: {real_avg:5.1f}% "
                f"(n={mask.sum():,})"
            )


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando WeatherRiskProbNet en dispositivo: {device}")

    labels, doy_idx, years, lats, lons, clim_table_norm = load_daily_dataset_and_context()

    # División estricta por años completos
    train_mask = years <= TRAIN_END_YEAR
    val_mask = (years > TRAIN_END_YEAR) & (years <= VAL_END_YEAR)
    test_mask = years > VAL_END_YEAR

    train_ds = DailyRiskDataset(
        labels[train_mask], doy_idx[train_mask], years[train_mask],
        lats, lons, clim_table_norm,
        num_samples=SAMPLES_PER_EPOCH, deterministic=False, seed=SEED,
    )
    val_ds = DailyRiskDataset(
        labels[val_mask], doy_idx[val_mask], years[val_mask],
        lats, lons, clim_table_norm,
        num_samples=VAL_SAMPLES, deterministic=True, seed=SEED + 1,
    )
    test_ds = DailyRiskDataset(
        labels[test_mask], doy_idx[test_mask], years[test_mask],
        lats, lons, clim_table_norm,
        num_samples=TEST_SAMPLES, deterministic=True, seed=SEED + 2,
    )

    pin_mem = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=pin_mem)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=pin_mem)

    model = WeatherRiskProbNet(in_channels=7, meta_dim=10, out_probs=3).to(device)
    # BCEWithLogitsLoss sin pos_weight para garantizar probabilidades reales calibradas
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        pbar = tqdm(train_loader, desc=f"Época {epoch:02d}/{EPOCHS}", leave=False)
        for x_patch, x_meta, y in pbar:
            x_patch = x_patch.to(device, non_blocking=True)
            x_meta = x_meta.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x_patch, x_meta)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * y.size(0)
            seen += y.size(0)
            pbar.set_postfix(loss=f"{running_loss / seen:.4f}")

        scheduler.step()
        val_metrics, _, _ = evaluate_probabilistic(model, val_loader, criterion, device)
        train_loss = running_loss / seen
        pt = val_metrics["per_target"]

        print(
            f"Época {epoch:02d}/{EPOCHS} | "
            f"Train BCE: {train_loss:.4f} | Val BCE: {val_metrics['loss']:.4f} | "
            f"Brier [Lluvia: {pt['Lluvia']['brier']:.3f}, "
            f"Viento: {pt['Viento Fuerte']['brier']:.3f}, "
            f"TempAdv: {pt['Temperaturas Adversas']['brier']:.3f}]"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                    "epoch": epoch,
                },
                BEST_MODEL_PATH,
            )

    # Verificación final sobre años de prueba inéditos (2020-2022)
    ckpt = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics, test_probs, test_targets = evaluate_probabilistic(model, test_loader, criterion, device)

    print("\n================ VERIFICACIÓN FINAL EN TEST (2020-2022) ================")
    print(f"Test BCE Loss: {test_metrics['loss']:.4f}")
    for name, stats in test_metrics["per_target"].items():
        print(
            f"  {name:22s} | Brier: {stats['brier']:.4f} | ROC-AUC: {stats['roc_auc']:.4f} | "
            f"Media Predicha: {stats['mean_pred_pct']:.1f}% vs Real: {stats['real_freq_pct']:.1f}%"
        )
    print_calibration_report(test_probs, test_targets)


if __name__ == "__main__":
    main()

