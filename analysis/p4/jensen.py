import sys, struct, numpy as np
from scipy.stats import binom
def read_labels(p): return np.loadtxt(p,dtype=np.int64)
def read_gt_ids(p):
    with open(p,'rb') as f:
        n,k=struct.unpack('<ii',f.read(8))
        return np.frombuffer(f.read(n*k*4),dtype=np.uint32).reshape(n,k)
datadir,f9,beta,k,outc = sys.argv[1],sys.argv[2],float(sys.argv[3]),int(sys.argv[4]),sys.argv[5]
meas={}
for line in open(f9):
    p=line.strip().split(',')
    if len(p)>=5 and p[0]=='clustered':
        try: meas[(int(p[1]),int(p[3]))]=float(p[4])/100.0
        except: pass
rows=[]
for T in [6,24,96]:
    lab=read_labels(f'{datadir}/lab_clustered{T}.txt'); qlab=read_labels(f'{datadir}/qlab_clustered{T}.txt')
    uids=read_gt_ids(f'{datadir}/unf_clustered{T}.bin')[:,:100]
    sig=(lab[uids]==qlab[:,None]).mean(1); np.save(f'{datadir}/sigq_clustered{T}.npy',sig)
    sm=sig.mean()
    for L in [75,150,300,600,1000]:
        n=max(int(round(beta*L)),k)
        hq=float((1-binom.cdf(k-1,n,sig)).mean()); hm=float(1-binom.cdf(k-1,n,sm))
        m=meas.get((T,L),float('nan')); dfc=hm-m
        ab=(hm-hq)/dfc if abs(dfc)>1e-9 else float('nan')
        rows.append((T,L,m*100,hm*100,hq*100,dfc*100,ab))
with open(outc,'w') as f:
    f.write('T,L,measured,h_mean,h_qavg,deficit_pp,absorbed\n')
    for r in rows: f.write(f'{r[0]},{r[1]},{r[2]:.3f},{r[3]:.3f},{r[4]:.3f},{r[5]:.3f},{r[6]:.4f}\n')
for r in rows: print(f'T={r[0]:>2} L={r[1]:>4} meas={r[2]:6.2f} h_mean={r[3]:6.2f} h_qavg={r[4]:6.2f} deficit={r[5]:6.2f}pp absorbed={r[6]:.3f}')
ab=np.array([r[6] for r in rows],float)
print(f'--- MEAN absorbed={np.nanmean(ab):.3f} spread=[{np.nanmin(ab):.3f},{np.nanmax(ab):.3f}] beta={beta} k={k}')
for T in [6,24,96]:
    s=np.load(f'{datadir}/sigq_clustered{T}.npy'); print(f'  sigma_loc T={T}: mean={s.mean():.4f} p10={np.quantile(s,.1):.4f} p50={np.median(s):.4f} p90={np.quantile(s,.9):.4f}')
