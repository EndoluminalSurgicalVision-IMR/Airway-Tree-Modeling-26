"""Dependency-free organizer scoring; all IDs, including zero, are real classes."""
import csv,json,re
from pathlib import Path
from statistics import mean

FRAME=re.compile(r'[0-9]+\.png')
IDENTIFIER=re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}')

def safe_id(value):
    if not isinstance(value,str) or not IDENTIFIER.fullmatch(value):raise ValueError('Unsafe identifier')
    return value

def rows(path,allowed,expected=None):
    if path.is_symlink() or not path.is_file() or path.stat().st_size>16*1024**2:raise ValueError('Invalid CSV file')
    with path.open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=['filename','label']:raise ValueError('CSV columns must be filename,label')
        data=list(reader)
    if any(set(r)!={'filename','label'} or not isinstance(r['filename'],str) or not isinstance(r['label'],str) or not re.fullmatch(r'[0-9]+',r['label']) for r in data):
        raise ValueError('Malformed CSV row; exactly two fields and an integer label required')
    names=[r['filename'] for r in data]
    if not names or len(set(names))!=len(names) or any(not FRAME.fullmatch(n) for n in names):raise ValueError('Invalid or duplicate frame names')
    if len({int(Path(n).stem) for n in names})!=len(names):raise ValueError('Duplicate numeric frame IDs')
    if names!=sorted(names,key=lambda n:int(Path(n).stem)):raise ValueError('Frame CSV rows are out of order')
    if expected is not None and names!=expected:raise ValueError('Missing, extra or mismatched frames')
    labels=[int(r['label']) for r in data]
    if not set(labels)<=allowed:raise ValueError('Unknown label ID')
    return names,labels

def segments(values):
    output=[];start=0
    for i in range(1,len(values)+1):
        if i==len(values) or values[i]!=values[start]:
            output.append((values[start],start,i));start=i
    return output

def edit_score(reference,prediction):
    a=[x[0] for x in segments(reference)];b=[x[0] for x in segments(prediction)]
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return 1-previous[-1]/max(len(a),len(b),1)

def segment_f1(reference,prediction,threshold=.5):
    gt=segments(reference);pred=segments(prediction);hits=set();tp=0
    for label,start,end in pred:
        candidates=[]
        for j,(true_label,left,right) in enumerate(gt):
            intersection=max(0,min(end,right)-max(start,left))
            iou=intersection/(max(end,right)-min(start,left)) if label==true_label else 0
            candidates.append((iou,-j,j))
        overlap,_,match=max(candidates)
        if overlap>=threshold and match not in hits:hits.add(match);tp+=1
    fp=len(pred)-tp;fn=len(gt)-tp
    return 2*tp/(2*tp+fp+fn),{'tp':tp,'fp':fp,'fn':fn}

def frame_scores(matrix):
    support=[sum(r) for r in matrix];predicted=[sum(r[i] for r in matrix) for i in range(len(matrix))]
    per_class=[]
    for i in range(len(matrix)):
        tp=matrix[i][i];precision=tp/predicted[i] if predicted[i] else 0;recall=tp/support[i] if support[i] else 0
        f1=2*tp/(support[i]+predicted[i]) if support[i]+predicted[i] else 0
        per_class.append({'support':support[i],'precision':precision,'recall':recall,'f1':f1})
    total=sum(support);present=[r for r in per_class if r['support']]
    return {'frame_accuracy':sum(matrix[i][i] for i in range(len(matrix)))/total,
        'frame_macro_f1':mean(r['f1'] for r in per_class),
        'macro_f1_present':mean(r['f1'] for r in present),
        'balanced_accuracy':mean(r['recall'] for r in present),
        'weighted_f1':sum(r['f1']*r['support'] for r in per_class)/total},per_class

def evaluate(predictions,ground_truth,schema,transition_graph=None):
    manifest=json.loads((ground_truth/'manifest.json').read_text(encoding='utf-8'))
    ids=[r['id'] for r in schema['labels']];allowed=set(ids);index={v:i for i,v in enumerate(ids)};n=len(ids)
    if manifest['label_ids']!=ids:raise ValueError('Ground truth schema differs')
    expected={safe_id(r['id'])+'.csv' for r in manifest['sequences']}
    if len(expected)!=len(manifest['sequences']) or not expected:raise ValueError('Duplicate/empty private manifest')
    if not predictions.is_dir() or {p.name for p in predictions.iterdir()}!=expected:raise ValueError('Missing/extra output files or directories')
    legal=None
    if transition_graph:
        if transition_graph.get('reviewed') is not True:raise ValueError('Only organizer-reviewed transition graphs may be used')
        legal={(v,v) for v in ids}
        for a,b in transition_graph['edges']:
            if a not in allowed or b not in allowed:raise ValueError('Unknown topology node')
            legal.add((a,b))
            if not transition_graph.get('directed',False):legal.add((b,a))
    matrix=[[0]*n for _ in ids];patients={};sequence_results=[];changes=adjacent=illegal=0
    for record in manifest['sequences']:
        sequence=safe_id(record['id']);patient=safe_id(record['patient'])
        names,target=rows(ground_truth/(sequence+'.csv'),allowed,record['frames'])
        _,pred=rows(predictions/(sequence+'.csv'),allowed,names)
        local=[[0]*n for _ in ids]
        for a,b in zip(target,pred):local[index[a]][index[b]]+=1;matrix[index[a]][index[b]]+=1
        scores,_=frame_scores(local);scores['sequence_edit']=edit_score(target,pred)
        scores['segment_f1_50'],counts=segment_f1(target,pred)
        change=sum(a!=b for a,b in zip(pred,pred[1:]));bad=sum((a,b) not in legal for a,b in zip(pred,pred[1:])) if legal is not None else None
        adjacent+=len(pred)-1;changes+=change
        if bad is not None:illegal+=bad
        sequence_results.append({'sequence':sequence,'patient':patient,'frames':len(pred),'scores':scores,
            'segment_counts':counts,'predicted_changes':change,'reference_changes':len(segments(target))-1,
            'illegal_transitions':bad})
        data=patients.setdefault(patient,{'matrix':[[0]*n for _ in ids],'sequences':[]})
        for i in range(n):
            for j in range(n):data['matrix'][i][j]+=local[i][j]
        data['sequences'].append(scores)
    aggregates,per_class=frame_scores(matrix);patient_results=[]
    for patient,data in sorted(patients.items()):
        scores,_=frame_scores(data['matrix'])
        scores.update(sequence_edit=mean(s['sequence_edit'] for s in data['sequences']),
                      segment_f1_50=mean(s['segment_f1_50'] for s in data['sequences']))
        patient_results.append({'patient':patient,'scores':scores,'sequences':len(data['sequences'])})
    aggregates.update(sequence_edit=mean(p['scores']['sequence_edit'] for p in patient_results),
        segment_f1_50=mean(p['scores']['segment_f1_50'] for p in patient_results),
        patient_mean_frame_accuracy=mean(p['scores']['frame_accuracy'] for p in patient_results),
        patient_mean_macro_f1_present=mean(p['scores']['macro_f1_present'] for p in patient_results))
    return {'valid':True,'aggregates':aggregates,'patients':len(patients),'sequences':len(sequence_results),
        'frames':sum(sum(r) for r in matrix),'per_class':[{'label':label,**r} for label,r in zip(ids,per_class)],
        'confusion_matrix':matrix,'results':patient_results,'sequence_results':sequence_results,
        'transition_diagnostics':{'predicted_changes':changes,'adjacent_frames':adjacent,
            'illegal_count':illegal if legal is not None else None,
            'illegal_transition_rate':illegal/adjacent if legal is not None and adjacent else None,
            'illegal_changed_transition_rate':illegal/changes if legal is not None and changes else None,
            'topology_supplied':legal is not None}}
