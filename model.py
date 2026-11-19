import torch
import torch.nn as nn


class WeatherRiskProbNet(nn.Module):
    def __init__(self, in_channels: int = 7, meta_dim: int = 10, out_probs: int = 3):
        super().__init__()

        # Bloque 1: Extractor espacial climatológico 2D (Capas 1, 2 y 3)
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

        # Bloque 2: Codificador esférico, estacional y de tendencia multianual (Capas 4 y 5)
        self.meta_encoder = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

        # Bloque 3: Fusión residual profunda (Capas 6, 7 y 8)
        self.fusion_in = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.fusion_res = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.fusion_compress = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

        # Bloque 4: Cabezal probabilístico multi-etiqueta (Capa 9)
        self.output_head = nn.Linear(64, out_probs)

    def forward(self, x_spatial_clim: torch.Tensor, x_coord_time: torch.Tensor) -> torch.Tensor:
        """
        x_spatial_clim: (Batch, 7, 5, 5)
        x_coord_time:   (Batch, 10)
        Retorna logits de forma (Batch, 3):
          [:, 0] -> logit de probabilidad de lluvia
          [:, 1] -> logit de probabilidad de viento fuerte
          [:, 2] -> logit de probabilidad de temperaturas adversas
        """
        spatial_vec = self.spatial_encoder(x_spatial_clim).flatten(start_dim=1)
        meta_vec = self.meta_encoder(x_coord_time)
        h = self.fusion_in(torch.cat([spatial_vec, meta_vec], dim=-1))
        h = h + self.fusion_res(h)
        h = self.fusion_compress(h)
        return self.output_head(h)

    @torch.no_grad()
    def predict_percentages(
        self, x_spatial_clim: torch.Tensor, x_coord_time: torch.Tensor
    ) -> torch.Tensor:
        """Devuelve probabilidades calibradas directamente en escala [0.0, 100.0]."""
        logits = self.forward(x_spatial_clim, x_coord_time)
        return torch.sigmoid(logits) * 100.0

