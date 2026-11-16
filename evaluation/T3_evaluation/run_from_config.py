"""Editable HOST paths; the participant receives no arguments and no ground truth."""
import argparse,json,subprocess,sys
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8'))
    command=[sys.executable,'-B',str(Path(__file__).with_name('run_submission.py')),
             '--image',config['image'],'--docker',config.get('docker','docker'),
             '--device',config.get('device','cuda'),'--timeout',str(config.get('timeout',21600))]
    for key,flag in [('input_root','--input-root'),('predictions_root','--predictions-root'),('gt_root','--gt-root'),('metrics_root','--metrics-root')]:
        path=Path(config[key]).expanduser()
        if not path.is_absolute():path=args.config.resolve().parent/path
        command.extend([flag,str(path)])
    raise SystemExit(subprocess.call(command))

if __name__=='__main__':main()
