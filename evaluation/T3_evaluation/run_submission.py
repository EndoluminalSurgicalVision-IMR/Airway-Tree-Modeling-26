"""Run a participant Docker, then score on the host; no split-specific branches."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from evaluate import score

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--docker',default='docker');parser.add_argument('--image',default='atm26-track3-baseline-batch:v3')
    parser.add_argument('--input-root',type=Path,required=True)
    parser.add_argument('--predictions-root',type=Path,required=True)
    parser.add_argument('--gt-root',type=Path,required=True)
    parser.add_argument('--metrics-root',type=Path,required=True)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--timeout',type=int,default=21600)
    parser.add_argument('--algorithm-args',nargs=argparse.REMAINDER,default=[])
    args=parser.parse_args()
    inp,pred,gt,report=[p.resolve() for p in [args.input_root,args.predictions_root,args.gt_root,args.metrics_root]]
    if not (inp/'sequences').is_dir():raise ValueError('input-root must contain sequences/')
    if {p.name for p in inp.iterdir()}!={'sequences'}:raise ValueError('input-root may contain only sequences/, no GT or metadata')
    if not (gt/'manifest.json').is_file():raise ValueError('gt-root must contain manifest.json')
    paths=[inp,pred,gt,report]
    if any(a==b or a.is_relative_to(b) or b.is_relative_to(a) for i,a in enumerate(paths) for b in paths[i+1:]):
        raise ValueError('Input, GT, predictions and metrics must be separate non-nested directories')
    if pred.exists() and any(pred.iterdir()):raise FileExistsError('Use a new/empty predictions-root')
    if report.exists() and any(report.iterdir()):raise FileExistsError('Use a new/empty metrics-root')
    pred.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    name='atm26-track3-host-'+str(time.time_ns());start=time.monotonic()
    command=[args.docker,'run','--rm','--name',name,'--platform','linux/amd64','--network','none','--read-only',
             '--cap-drop','ALL','--security-opt','no-new-privileges','--pids-limit','256','--memory','8g','--cpus','8',
             '--tmpfs','/tmp:rw,size=1073741824','--mount',f'type=bind,src={inp},dst=/input,readonly',
             '--mount',f'type=bind,src={pred},dst=/output']
    if args.device=='cuda':command+=['--gpus','all']
    command+=[args.image,*args.algorithm_args]
    state={'status':'running','algorithm_command':command,'private_gt_mounted_in_algorithm':False,
           'evaluation_runtime':'host_python','paths':dict(input=str(inp),predictions=str(pred),ground_truth=str(gt),metrics=str(report))}
    def save(): (report/'run.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        with (report/'algorithm.stdout.log').open('w',encoding='utf-8') as out,(report/'algorithm.stderr.log').open('w',encoding='utf-8') as err:
            process=subprocess.Popen(command,stdout=out,stderr=err)
            try:code=process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                subprocess.run([args.docker,'rm','-f',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                process.kill();process.wait();raise TimeoutError('Algorithm exceeded timeout')
        if code:raise RuntimeError(f'Algorithm exited {code}; see algorithm.stderr.log')
        result=score(pred,gt,report)
        if not result['valid']:raise ValueError(result['error'])
        state.update(status='complete',aggregates=result['aggregates'])
    except Exception as error:state.update(status='failed',error=str(error))
    state['seconds']=time.monotonic()-start;save()
    print(json.dumps(state,indent=2))
    return 0 if state['status']=='complete' else 1

if __name__=='__main__':raise SystemExit(main())
