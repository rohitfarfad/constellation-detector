import sys,importlib.util,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import constellation as c
import numpy as np
spec=importlib.util.spec_from_file_location('v4','outputs/q5/baseline_v4/constellation.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

rng=np.random.default_rng(107);result={}
for perspective in [False,True]:
 errors=[[],[]];changed=0;valid=[0,0]
 for k in range(150):
  nodes=rng.uniform(0,400,(14,2));matrix=np.array([[2.,-.4,500],[.4,2.,600],[0.,0.,1.]])
  if perspective:matrix[2,:2]=rng.uniform(-.00012,.00012,2)
  truth=c.cv2.perspectiveTransform(nodes[None],matrix)[0];observed=truth[:9]+rng.normal(0,.8,(9,2))
  fits=[old.independent_transform(nodes[:9],observed),c.independent_transform(nodes[:9],observed)]
  changed+=all(x is not None for x in fits) and not np.allclose(fits[0][0],fits[1][0])
  for j,fit in enumerate(fits):
   if fit is not None:
    e=np.linalg.norm(c.cv2.perspectiveTransform(nodes[9:][None],fit[0])[0]-truth[9:],axis=1)
    errors[j].extend(e.tolist());valid[j]+=1
 result['projective' if perspective else 'affine']={'changed_models':int(changed),'accepted':valid,'median_error':[float(np.median(x)) for x in errors],'p95_error':[float(np.percentile(x,95)) for x in errors],'within_6':[float(np.mean(np.array(x)<=6)) for x in errors]}
print(json.dumps(result,indent=2))
Path('outputs/q5/experiments/v5_geometry_final_synthetic.json').write_text(json.dumps(result,indent=2))
