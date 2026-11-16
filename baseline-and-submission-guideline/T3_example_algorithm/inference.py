"""IO demonstration only: constant legal label 0; replace build_predictor/predict_case."""
import argparse,csv,json,re,os
from pathlib import Path

def build_predictor():
    # A tiny JSON resource demonstrates model packaging; not a trained airway model.
    model=Path(__file__).resolve().parent/'resources/placeholder/model.json'
    return json.loads(model.read_text())['constant_label']

def predict_case(predictor,frame_paths):
    # Replace with your own model; return exactly one semantic label per frame.
    return [predictor]*len(frame_paths)

def main(single_case=True):
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,default=Path('/input'))
    parser.add_argument('--output',type=Path,default=Path('/output'));args=parser.parse_args()
    source=args.input/'sequences'
    if not source.is_dir():raise ValueError('Expected /input/sequences/<id>/*.png')
    jobs=sorted(source.iterdir())
    if not jobs or (single_case and len(jobs)!=1):raise ValueError('Expected '+('one sequence' if single_case else 'at least one sequence'))
    target=args.output/'predictions';target.mkdir(parents=True,exist_ok=True)
    if any(target.iterdir()):raise FileExistsError('Use an empty output directory')
    allowed={r['id'] for r in json.loads(Path(__file__).with_name('label_schema.json').read_text())['labels']}
    predictor=build_predictor()
    for job in jobs:
        if not job.is_dir() or job.is_symlink() or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',job.name):raise ValueError('Invalid sequence ID')
        frames=list(job.iterdir())
        if not frames or any(p.is_symlink() or not p.is_file() or not re.fullmatch(r'[0-9]+\.png',p.name) for p in frames):raise ValueError('Expected numeric PNG frames only')
        frames.sort(key=lambda p:int(p.stem))
        if len({int(p.stem) for p in frames})!=len(frames):raise ValueError('Duplicate numeric frame IDs')
        labels=predict_case(predictor,frames)
        if len(labels)!=len(frames) or any(type(v) is not int or v not in allowed for v in labels):raise ValueError('Invalid model output')
        output=target/(job.name+'.csv');temporary=output.with_suffix('.tmp')
        with temporary.open('x',newline='',encoding='utf-8') as stream:
            writer=csv.writer(stream);writer.writerow(['filename','label']);writer.writerows((p.name,v) for p,v in zip(frames,labels))
        os.replace(temporary,output)
    print(json.dumps({'sequences':len(jobs),'placeholder_predictions':True}))

if __name__=='__main__':main()
