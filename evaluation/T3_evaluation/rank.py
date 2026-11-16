"""Equal-weight mean ranks over the four organizer-selected metrics."""
import argparse,csv,json,math
from pathlib import Path
from scoring import safe_id

def mean_ranks(values,decimals=12,descending=True):
    ordered=sorted(range(len(values)),key=lambda i:round(values[i],decimals),reverse=descending);ranks=[0.]*len(values);start=0
    while start<len(ordered):
        end=start+1
        while end<len(ordered) and round(values[ordered[end]],decimals)==round(values[ordered[start]],decimals):end+=1
        value=(start+1+end)/2
        for position in range(start,end):ranks[ordered[position]]=value
        start=end
    return ranks

def leaderboard(reports,policy):
    valid=[];invalid=[];seen=set()
    for name,report in reports:
        safe_id(name)
        if name in seen:raise ValueError('Duplicate submission ID')
        seen.add(name)
        if not report.get('valid'):invalid.append({'submission':name,'status':'invalid','rank':None});continue
        values=report['aggregates']
        if any(not isinstance(values.get(k),(int,float)) or not math.isfinite(values[k]) or not 0<=values[k]<=1 for k in policy['metrics']):
            raise ValueError('Missing or invalid ranking metric')
        valid.append({'submission':name,'status':'valid',**{k:values[k] for k in policy['metrics']}})
    for metric in policy['metrics']:
        ranks=mean_ranks([row[metric] for row in valid],policy['rounding_decimals_for_ties'])
        for row,rank in zip(valid,ranks):row[metric+'_rank']=rank
    for row in valid:row['mean_rank']=sum(row[k+'_rank'] for k in policy['metrics'])/len(policy['metrics'])
    valid.sort(key=lambda r:(r['mean_rank'],r['submission']))
    for i,row in enumerate(valid):
        row['rank']=valid[i-1]['rank'] if i and row['mean_rank']==valid[i-1]['mean_rank'] else i+1
    return valid+invalid

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reports',type=Path,required=True,help='Directory containing <submission-id>.json evaluator outputs')
    parser.add_argument('--policy',type=Path,default=Path(__file__).with_name('policy.json'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();policy=json.loads(args.policy.read_text())
    rows=leaderboard([(p.stem,json.loads(p.read_text())) for p in sorted(args.reports.glob('*.json'))],policy)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'leaderboard.json').write_text(json.dumps({'policy':policy,'results':rows},indent=2)+'\n')
    fields=['submission','status','rank','mean_rank',*policy['metrics'],*[k+'_rank' for k in policy['metrics']]]
    with (args.output/'leaderboard.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)

if __name__=='__main__':main()
