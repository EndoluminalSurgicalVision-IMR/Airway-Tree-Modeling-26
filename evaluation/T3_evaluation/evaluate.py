"""Host-side scoring. The identical command supports valid and test directories."""
import argparse,json,sys,traceback
from pathlib import Path
from scoring import evaluate

def score(predictions_root,gt_root,output_root,schema_path=None,transition_graph=None):
    output_root=Path(output_root);output_root.mkdir(parents=True,exist_ok=True)
    source=Path(predictions_root)
    try:
        if (source/'predictions').is_dir():
            if {p.name for p in source.iterdir()}!={'predictions'}:raise ValueError('Unexpected Docker output root files')
            source=source/'predictions'
        schema=json.loads(Path(schema_path or Path(__file__).with_name('label_schema.json')).read_text(encoding='utf-8'))
        graph=json.loads(Path(transition_graph).read_text(encoding='utf-8')) if transition_graph else None
        result=evaluate(source,Path(gt_root),schema,graph)
        result.update(evaluator_version='track3-host-v3',evaluation_runtime='host_python')
    except Exception as error:
        result={'valid':False,'error':str(error),'aggregates':{}}
    (output_root/'metrics.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions-root',type=Path,required=True)
    parser.add_argument('--gt-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--schema',type=Path);parser.add_argument('--transition-graph',type=Path)
    args=parser.parse_args()
    result=score(args.predictions_root,args.gt_root,args.output_root,args.schema,args.transition_graph)
    print(json.dumps({k:result[k] for k in ['valid','aggregates','error'] if k in result}))
    return 0 if result['valid'] else 1

if __name__=='__main__':raise SystemExit(main())
