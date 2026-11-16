"""Stage a local or downloaded checkpoint, with mandatory official SHA256 check."""
import argparse,hashlib,shutil,urllib.request
from pathlib import Path
DEFAULT_URL='https://huggingface.co/Endoluminal-Surg-Vision-IMR/Airway-Tree-Modeling-26-baselines/resolve/main/T3_baseline/resources/clip_mlp/checkpoint.pt'
EXPECTED='2cc20ae3ca7a0aa86bba708fc9e0d7da68fc78aff2f5d780ffc6790d514d04fe'
def digest(file):
    sha=hashlib.sha256()
    with file.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):sha.update(chunk)
    return sha.hexdigest()
def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group()
    group.add_argument('--url',default=DEFAULT_URL,help='Checkpoint URL; defaults to the published Hugging Face file')
    group.add_argument('--source',type=Path,help='Existing local checkpoint.pt')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'resources/clip_mlp/checkpoint.pt')
    args=parser.parse_args();dest=args.output
    if dest.exists():
        if digest(dest)!=EXPECTED:raise ValueError('Existing checkpoint differs; refusing to overwrite')
        print('Verified existing official checkpoint');return
    if not args.source and not args.url:parser.error('Specify --source or --url')
    dest.parent.mkdir(parents=True,exist_ok=True);temporary=dest.with_suffix('.download')
    if temporary.exists():raise FileExistsError(temporary)
    try:
        if args.source:shutil.copyfile(args.source,temporary)
        else:
            if not args.url.startswith(('https://','http://')):raise ValueError('Expected an HTTP(S) URL')
            with urllib.request.urlopen(args.url,timeout=60) as source,temporary.open('xb') as out:
                shutil.copyfileobj(source,out,1024**2)
        if digest(temporary)!=EXPECTED:raise ValueError('Downloaded checkpoint SHA256 mismatch')
        temporary.replace(dest);print('Official checkpoint ready:',dest)
    finally:temporary.unlink(missing_ok=True)
if __name__=='__main__':main()
