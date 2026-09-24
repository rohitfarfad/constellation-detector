"""Stress the joint reassignment with reciprocal errors constructed from real candidates."""
import sys,json,csv,ast,copy
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import constellation as c
import numpy as np
c.cv2.setNumThreads(1)
ev=json.load(open('outputs/q5/experiments/v6_evidence.json'))
base=json.load(open('outputs/q5/cross_validation_v5.json'))
patterns=c.read_patterns('Data/patterns');report={}
for row in csv.DictReader(open('Data/train_ground_truth.csv')):
 name=row['Id'];p=base[name]['prediction'];ms=ev[name];ps=[np.array(v) for v in p['probabilities']]
 image=c.cv2.imread(f'Data/train/{name}/{name}_image.png',0).astype(np.float32)
 stars=np.maximum.reduce([c.cv2.GaussianBlur(image,(0,0),s)-c.cv2.GaussianBlur(image,(0,0),6*s) for s in [2.,6.,18.]])
 points=[];owners=[]
 for i,(matches,probs) in enumerate(zip(ms,ps)):
  for j in np.argsort(probs)[::-1][:6]:
   x,y=matches[j]['x'],matches[j]['y']
   if probs[j]>=max(.001,.05*probs.max()) and stars[round(y),round(x)]>25:
    points.append([x,y]);owners.append(i)
 points=np.array(points);owners=np.array(owners);model=p['ranking'][0];nodes=patterns[model['name']]
 original=copy.deepcopy(p['patches'])
 c.resolve_figure_pairs(original,ms,ps,nodes,points,owners,model)
 assert original==p['patches']
 cases=corrected=unwanted=0;details=[]
 truth=[ast.literal_eval(row[f'patch_{i+1:02}']) for i in range(len(ms))]
 figures=[i for i,t in enumerate(truth) if t!=-1 and t[2] and p['patches'][i]!=-1 and p['patches'][i][2]]
 for pos,i in enumerate(figures):
  for k in figures[pos+1:]:
   a=np.array([[m['x'],m['y']] for m in ms[i]]);b=np.array([[m['x'],m['y']] for m in ms[k]])
   da=np.linalg.norm(a-truth[k][:2],axis=1);db=np.linalg.norm(b-truth[i][:2],axis=1)
   if da.min()>12 or db.min()>12:continue
   prediction=copy.deepcopy(p['patches']);prediction[i]=[*a[da.argmin()].tolist(),1];prediction[k]=[*b[db.argmin()].tolist(),1]
   before=copy.deepcopy(prediction)
   swaps=c.resolve_figure_pairs(prediction,ms,ps,nodes,points,owners,model)
   cases+=1
   if swaps:
    good=all(np.linalg.norm(np.array(prediction[q][:2])-truth[q][:2])<=12 for q in [i,k])
    other_changes=[q for q in range(len(ms)) if q not in [i,k] and prediction[q]!=before[q]]
    corrected+=good and not other_changes;unwanted+=not good or bool(other_changes)
    details.append({'queries':[i+1,k+1],'corrected':bool(good),'other_changes':other_changes})
 report[name]={'baseline_unchanged':True,'constructed_reciprocal_errors':cases,'corrected_pairs':corrected,'unwanted_changes':unwanted,'details':details}
 print(name,report[name],flush=True)
Path('outputs/q5/experiments/v6_pair_check.json').write_text(json.dumps(report,indent=2))
