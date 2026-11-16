"""Shared offline CLIP+MLP runtime. Independent raw frame argmax only."""
import json
import os
import re
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')

import torch
from torchvision.transforms import v2
from vendor.clip_mlp_model import CLIPMLPClassifier

SCHEMA = Path(__file__).with_name('label_schema.json')
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class FrameClassifier(CLIPMLPClassifier):
    def __init__(self, backbone_name, num_classes, hidden_dim, dropout, local_files_only,
                 initialize_from_config=False):
        if not initialize_from_config:
            super().__init__(backbone_name,num_classes,hidden_dim,dropout,local_files_only)
            return
        # A full trained checkpoint already contains every CLIP parameter.
        # Construct the same architecture without shipping/loading a second
        # unused 600 MB copy of the original pretrained weights.
        from transformers import AutoConfig,AutoModel
        torch.nn.Module.__init__(self)
        self.backbone_name=backbone_name
        self.backbone=AutoModel.from_config(AutoConfig.from_pretrained(backbone_name,local_files_only=True))
        dimension=int(self.backbone.config.vision_config.hidden_size)
        self.classifier=torch.nn.Sequential(torch.nn.LayerNorm(dimension),torch.nn.Linear(dimension,hidden_dim),
            torch.nn.GELU(),torch.nn.Dropout(dropout),torch.nn.Linear(hidden_dim,num_classes))

    def features(self, pixel_values):
        # Transformers 4.57's internal CLIPVisionTransformer no longer accepts
        # return_dict. It always returns BaseModelOutputWithPooling.
        return self._extract_features(self.backbone.vision_model(pixel_values=pixel_values))

    def forward(self, pixel_values):
        return self.classifier(self.features(pixel_values))


def label_ids():
    return [item['id'] for item in json.loads(SCHEMA.read_text(encoding='utf-8'))['labels']]


def check_labels(values):
    if values != label_ids():
        raise ValueError('Checkpoint label_ids must exactly match the current 32-class label_schema.json, including order. Old 27-class checkpoints are incompatible.')


def make_transform(image_size):
    # Exactly the source dataset.make_transform(..., training=False).
    return v2.Compose([v2.ToImage(), v2.Resize((image_size, image_size), antialias=True),
                       v2.ToDtype(torch.float32, scale=True), v2.Normalize(CLIP_MEAN, CLIP_STD)])


def device_of(value):
    if value == 'auto':
        value = 'cuda' if torch.cuda.is_available() else 'cpu'
    if value == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    return torch.device(value)


def local_resource(model_dir, name):
    root = Path(model_dir).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Model resources must stay inside model-dir')
    return path


def load_model(model_dir, device, *, require_checkpoint=True):
    config = json.loads((Path(model_dir) / 'model.json').read_text(encoding='utf-8'))
    check_labels(config['label_ids'])
    checkpoint = None
    if require_checkpoint:
        checkpoint = torch.load(local_resource(model_dir, config['checkpoint']), map_location='cpu', weights_only=True)
        check_labels(checkpoint.get('label_ids'))
        if checkpoint.get('image_size') is not None and checkpoint['image_size'] != config['image_size']:
            raise ValueError('Checkpoint preprocessing size differs from model.json')
    torch.manual_seed(config.get('seed', 17))
    model = FrameClassifier(
        backbone_name=str(local_resource(model_dir, config['backbone'])), num_classes=len(label_ids()),
        hidden_dim=config['mlp_hidden_dim'], dropout=config['dropout'], local_files_only=True,
        initialize_from_config=config.get('initialize_from_config',False))
    if checkpoint is not None:
        if config['checkpoint_kind'] == 'head':
            model.classifier.load_state_dict(checkpoint['classifier'], strict=True)
        elif config['checkpoint_kind'] == 'full':
            model.load_state_dict(checkpoint['model'], strict=True)
        else:
            raise ValueError('checkpoint_kind must be head or full')
    model.to(device).eval()
    return model, make_transform(config['image_size']), config


def sequences(input_dir):
    root = Path(input_dir) / 'sequences'
    if not root.is_dir():
        raise ValueError(f'Expected {root}')
    result = []
    for sequence in sorted(root.iterdir()):
        if not sequence.is_dir() or sequence.is_symlink() or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', sequence.name):
            raise ValueError(f'Invalid sequence directory: {sequence}')
        files = list(sequence.iterdir())
        if not files or any(not f.is_file() or f.is_symlink() or not re.fullmatch(r'[0-9]+\.png', f.name) for f in files):
            raise ValueError(f'Input sequences must contain only numbered PNG frames, without labels/metadata: {sequence}')
        files.sort(key=lambda p: int(p.stem))
        if len({int(f.stem) for f in files}) != len(files):
            raise ValueError('Duplicate numeric frame indices')
        result.append((sequence.name, files))
    if not result:
        raise ValueError('No sequences found')
    return result
