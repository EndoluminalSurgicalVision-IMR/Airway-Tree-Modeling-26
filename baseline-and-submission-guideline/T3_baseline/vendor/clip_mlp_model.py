from __future__ import annotations

import torch
from torch import nn


class CLIPMLPClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        hidden_dim: int,
        dropout: float,
        local_files_only: bool,
    ):
        super().__init__()
        from transformers import AutoModel

        self.backbone_name = backbone_name
        self.backbone = AutoModel.from_pretrained(
            backbone_name,
            local_files_only=local_files_only,
        )
        if hasattr(self.backbone.config, "vision_config"):
            feature_dim = int(self.backbone.config.vision_config.hidden_size)
        elif hasattr(self.backbone.config, "projection_dim"):
            feature_dim = int(self.backbone.config.projection_dim)
        else:
            feature_dim = int(self.backbone.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "vision_model"):
            outputs = self.backbone.vision_model(pixel_values=pixel_values, return_dict=True)
            features = self._extract_features(outputs)
        elif hasattr(self.backbone, "get_image_features"):
            features = self.backbone.get_image_features(pixel_values=pixel_values)
        else:
            outputs = self.backbone(pixel_values=pixel_values, return_dict=True)
            features = self._extract_features(outputs)
        features = self._extract_features(features)
        return self.classifier(features)

    @staticmethod
    def _extract_features(outputs) -> torch.Tensor:
        if torch.is_tensor(outputs):
            return outputs
        features = getattr(outputs, "image_embeds", None)
        if features is not None:
            return features
        features = getattr(outputs, "pooler_output", None)
        if features is not None:
            return features
        features = getattr(outputs, "last_hidden_state", None)
        if features is not None:
            return features[:, 0]
        if isinstance(outputs, (tuple, list)) and outputs and torch.is_tensor(outputs[0]):
            return outputs[0]
        raise TypeError(f"Could not extract tensor features from {type(outputs).__name__}.")

    def trainable_parameter_counts(self) -> tuple[int, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return trainable, total
