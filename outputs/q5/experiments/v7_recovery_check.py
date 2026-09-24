import sys,json,csv,copy,ast,tempfile
from pathlib import Path
sys.path.insert(0,str(Path.cwd()));sys.path.insert(0,str(Path('outputs/q5/experiments').resolve()))
import constellation as c
import v7_recovery_function as trial
import importlib.util
spec=importlib.util.spec_from_file_location('v6_control','outputs/q5/baseline_v6/constellation.py');control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)
import numpy as np
c.cv2.setNumThreads(1)
base={n:r['prediction'] for n,r in json.load(open('outputs/q5/cross_validation_v6.json')).items()};base.update(json.load(open('outputs/q5/competition_diagnostics_v6.json')))
ev=json.load(open('outputs/q5/experiments/v6_evidence.json'));contexts=json.load(open('outputs/q5/experiments/v7_contexts.json'));patterns=c.read_patterns('Data/patterns');results={}
for name,p in base.items():
 ctx=contexts[name];model=p['ranking'][0];pred=copy.deepcopy(p['patches'])
 recovered=trial.recover_figure_queries(pred,ev[name],[np.array(x) for x in p['probabilities']],patterns[model['name']],np.array(ctx['points']),np.array(ctx['owners']),model)
 changes=[dict(query=i+1,before=a,after=b) for i,(a,b) in enumerate(zip(p['patches'],pred)) if a!=b]
 results[name]={'changes':changes,'recoveries':recovered}
 if changes:print(name,results[name],flush=True)
Path('outputs/q5/experiments/v7_recovery_results.json').write_text(json.dumps(results,indent=2))
rows=list(csv.DictReader(open('Data/train_ground_truth.csv')));rng=np.random.default_rng(117);temp=Path(tempfile.mkdtemp(prefix='q5-v7-stress-'))/'query.png';checks=[]; tested=low_confidence=absent_tests=0
for row in rows:
 name=row['Id'];p=base[name];ctx=contexts[name];model=p['ranking'][0];nodes=patterns[model['name']];points=np.array(ctx['points']);owners=np.array(ctx['owners']);cal=c.fit_calibration(*c.calibration_data([r for r in rows if r!=row],ev))
 image=c.cv2.imread(f'Data/train/{name}/{name}_image.png',0)
 for i,matches in enumerate(ev[name]):
  truth=ast.literal_eval(row[f'patch_{i+1:02}']);q=c.cv2.imread(f'Data/train/{name}/patches/patch_{i+1:02}.png',0).astype(float)
  for sigma in [5,15,25]:
   query=np.clip(q+rng.normal(0,sigma,q.shape),0,255).astype(np.uint8);ms=[]
   for m in matches:
    match=c.match_features(image,query,copy.deepcopy(m))
    if match is not None:ms.append(match)
   if not ms:continue
   tested+=1
   c.cv2.imwrite(str(temp),query);c.residual_evidence(image,[temp],[ms]);ps=c.candidate_probabilities(ms,cal)
   evidence=list(ev[name]);evidence[i]=ms;probs=[np.array(x) for x in p['probabilities']];probs[i]=ps
   # Isolate low-confidence recovery, not already accepted image matches.
   if ps.max()>=.04:continue
   low_confidence+=1;absent_tests+=truth==-1
   before=copy.deepcopy(p['patches']);before[i]=-1;after=copy.deepcopy(before)
   a=control.recover_figure_queries(before,evidence,probs,nodes,points,owners,model)
   b=trial.recover_figure_queries(after,evidence,probs,nodes,points,owners,model)
   if before[i]!=after[i]:
    correct=truth!=-1 and np.linalg.norm(np.array(after[i][:2])-truth[:2])<=12
    checks.append(dict(scene=name,query=i+1,sigma=sigma,truth=truth,before=before[i],after=after[i],correct=bool(correct)))
 print('noise check',name,'new decisions',len(checks),flush=True)
Path('outputs/q5/experiments/v7_recovery_noise_check.json').write_text(json.dumps({'seed':117,'noise_sigmas':[5,15,25],'valid_query_variants':tested,'low_confidence_variants_checked':low_confidence,'absent_variants_checked':absent_tests,'new_decisions':checks},indent=2))
print('New decisions',checks,flush=True)
