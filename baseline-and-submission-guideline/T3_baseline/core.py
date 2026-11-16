"""Proposed Track-3 batch I/O: PNG sequences -> filename,label CSV files."""
import argparse
import csv
import json
import os
from pathlib import Path
import time

import torch
from PIL import Image
from runtime import device_of, label_ids, load_model, sequences


def predict_sequences(model, transform, input_dir, output_dir, device, batch_size=8):
    if batch_size < 1:
        raise ValueError('batch-size must be positive')
    jobs = sequences(input_dir)
    output = Path(output_dir) / 'predictions'
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Use a new or empty predictions directory to avoid stale output')
    ids = label_ids()
    total = 0
    with torch.inference_mode():
        for sequence_id, files in jobs:
            started = time.perf_counter()
            destination = output / f'{sequence_id}.csv'
            temporary = destination.with_suffix('.tmp')
            try:
                with temporary.open('x', encoding='utf-8', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['filename', 'label'])
                    for offset in range(0, len(files), batch_size):
                        batch_files = files[offset:offset + batch_size]
                        images = []
                        for file in batch_files:
                            with Image.open(file) as image:
                                images.append(transform(image.convert('RGB')))
                        pixels = torch.stack(images).to(device)
                        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                            logits = model(pixels)
                        if logits.shape != (len(batch_files), len(ids)) or not torch.isfinite(logits).all():
                            raise ValueError('Model returned invalid frame logits')
                        predicted = logits.argmax(dim=-1).cpu().tolist()
                        writer.writerows((file.name, ids[index]) for file,index in zip(batch_files,predicted))
                        total += len(batch_files)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            print(json.dumps(dict(sequence=sequence_id, frames=len(files), seconds=round(time.perf_counter()-started,3))), flush=True)
            del images, pixels, logits
            if device.type == 'cuda': torch.cuda.empty_cache()
    return dict(sequences=len(jobs), frames=total, decode='raw_argmax', viterbi=False)


def main(single_case=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('/input'))
    parser.add_argument('--output', type=Path, default=Path('/output'))
    parser.add_argument('--model-dir', type=Path, default=Path(os.environ.get('T3_MODEL_DIR',str(Path(__file__).resolve().parent/'resources/clip_mlp'))))
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    torch.set_num_threads(args.threads)
    if single_case and len(sequences(args.input)) != 1:
        parser.error("Per-case baseline requires exactly one sequence; use the batch version for many")
    device = device_of(args.device)
    model, transform, config = load_model(args.model_dir, device)
    print(json.dumps(dict(device=str(device), training_status=config['training_status'], classes=len(label_ids()))), flush=True)
    print(json.dumps(predict_sequences(model,transform,args.input,args.output,device,args.batch_size)), flush=True)


if __name__ == '__main__':
    main()
