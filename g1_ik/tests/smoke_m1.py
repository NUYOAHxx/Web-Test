import numpy as np, time, sys
from g1ik.config import load_config
from g1ik.solver import IKSolver, IKRequest
cfg=load_config(); S=IKSolver(cfg); r=S.robot
rng=np.random.default_rng(1)
lo,hi=r.lower,r.upper
n=int(sys.argv[1]) if len(sys.argv)>1 else 8
ok=0; tt=[]
for k in range(n):
  while True:
    q=r.q_nominal.copy(); ia=r.arm_idx['left']; q[ia]=lo[ia]+(hi[ia]-lo[ia])*rng.uniform(0.15,0.85,7)
    if S.coll.mesh_check(q)[1]>0.02: break
  T=r.fk(q)['left']
  res=S.solve(IKRequest(targets={'left':T}, mode='M1'))
  e=res['log'][0]; s=res['solutions'][0]
  ok+=res['success']; tt.append(res['time'])
  print(k,res['success'],'t %.2f'%res['time'],{a:round(b,2) for a,b in e['times'].items()},'pos %.1e rot %.1e util %.2f d %.3f'%(s['pos_err'],s['rot_err'],s['max_util'],s['min_mesh_dist']),s['fail'],s['nlp_status'],s['nlp_iters'])
print('success',ok,'/',n,'median t',np.median(tt),'nlp builds',S.nlp.n_build,S.nlp.build_time)
