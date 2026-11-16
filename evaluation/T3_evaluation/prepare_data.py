"""Prepare image-only Docker input and private GT from a curated split root."""
import argparse,csv,hashlib,hmac,json,os,secrets
from pathlib import Path

def prepare(data_root,input_root,gt_root,demo=False):
    source,public,private=[Path(p).resolve() for p in [data_root,input_root,gt_root]]
    paths=[source,public,private]
    if any(a==b or a.is_relative_to(b) or b.is_relative_to(a) for i,a in enumerate(paths) for b in paths[i+1:]):
        raise ValueError('Source, input and GT must be separate non-nested directories')
    if public.exists() or private.exists():raise FileExistsError('Use new input and GT directories')
    schema=json.loads((Path(__file__).with_name('label_schema.json')).read_text())
    ids=[r['id'] for r in schema['labels']];secret=secrets.token_bytes(32)
    def opaque(kind,value):return kind+'_'+hmac.new(secret,value.encode(),hashlib.sha256).hexdigest()[:20]
    cases=sorted(p for p in source.iterdir() if p.is_dir())
    if demo:cases=cases[:1]
    public.mkdir(parents=True);private.mkdir(parents=True)
    records=[];provenance=[]
    for case in cases:
        index=json.loads((case/'index.json').read_text(encoding='utf-8'))
        routes=sorted(index['paths'],key=lambda r:int(r['folder']))
        if demo:routes=routes[:2]
        for route in routes:
            folder=case/str(route['folder']);files=sorted(folder.glob('*.png'),key=lambda f:int(f.stem))
            if demo and len(files)>8:files=[files[round(i*(len(files)-1)/7)] for i in range(8)]
            if not files:raise ValueError('Empty source sequence: '+str(folder))
            with (folder/'labels.csv').open(encoding='utf-8-sig',newline='') as stream:
                truth={r['filename']:int(r['label']) for r in csv.DictReader(stream)}
            identity=opaque('seq',case.name+'/'+folder.name);patient=opaque('case',case.name)
            target=public/'sequences'/identity;target.mkdir(parents=True)
            for file in files:
                if truth[file.name] not in ids:raise ValueError('Unknown GT label')
                os.link(file,target/file.name)
            with (private/(identity+'.csv')).open('w',newline='',encoding='utf-8') as stream:
                writer=csv.writer(stream);writer.writerow(['filename','label']);writer.writerows((f.name,truth[f.name]) for f in files)
            records.append({'id':identity,'patient':patient,'frames':[f.name for f in files]})
            provenance.append({'id':identity,'source_case':case.name,'source_folder':folder.name})
    if not records:raise ValueError('No sequences found')
    manifest={'label_ids':ids,'sequences':records}
    (private/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (private/'provenance.json').write_text(json.dumps({'secret_hex':secret.hex(),'sources':provenance},indent=2)+'\n')
    return {'patients':len(cases),'sequences':len(records),'frames':sum(len(r['frames']) for r in records)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True);parser.add_argument('--input-root',type=Path,required=True)
    parser.add_argument('--gt-root',type=Path,required=True);parser.add_argument('--demo',action='store_true')
    args=parser.parse_args();print(json.dumps(prepare(args.data_root,args.input_root,args.gt_root,args.demo)))

if __name__=='__main__':main()
