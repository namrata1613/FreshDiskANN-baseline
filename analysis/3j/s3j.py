#!/usr/bin/env python3
"""
3j sigma_loc sweep: calibrate h(sigma_loc, L, k) = Pr[Binom(L, sigma_loc) >= k].

Measures the *algorithmic* post-filter recall (post-filter-AFTER): the base graph
search returns the L nearest (unfiltered); we filter to the query's own label and
take the top-k; recall@k vs per-label GT. This is exactly the theory h object and
matches C2_DISK_POSTFILTER (disk post-filter of results). One label-blind base
graph serves every (sigma_loc, L) config.

Subcommands:
  genlabels  base_bin N T mode out_labels out_qlabels query_bin   # mode: diffuse|clustered
  gt         base_bin query_bin labels qlabels N k out_perlabel out_unfilt M
  sigmaloc   qlabels labels_of_base unfilt_gt M out_row
  pfrecall   restags labels_of_base qlabels perlabel_gt k out_row  # restags = search dump n x K uint32
  fit        recall_csv out_prefix

.bin formats (DiskANN, little-endian):
  vecs: int32 n, int32 d, n*d float32
  gt (2-block): int32 n, int32 k, n*k uint32 ids, n*k float32 dists   (K pinned; NOT 3-block-with-tags)
  search result tags: int32 n, int32 K, n*K uint32
labels/qlabels: one int per line (matches 2E_labels.txt).
"""
import sys, struct
import numpy as np

def read_vecs(path):
    with open(path,'rb') as f:
        n,d=struct.unpack('<ii',f.read(8))
        a=np.frombuffer(f.read(n*d*4),dtype=np.float32).reshape(n,d)
    return a

def read_labels(path):
    return np.loadtxt(path,dtype=np.int64)

def write_gt(path, ids, dists):
    n,k=ids.shape
    with open(path,'wb') as f:
        f.write(struct.pack('<ii',n,k))
        f.write(ids.astype(np.uint32).tobytes())
        f.write(dists.astype(np.float32).tobytes())

def read_gt_ids(path):
    with open(path,'rb') as f:
        n,k=struct.unpack('<ii',f.read(8))
        ids=np.frombuffer(f.read(n*k*4),dtype=np.uint32).reshape(n,k)
    return ids

def read_tags(path):
    with open(path,'rb') as f:
        n,k=struct.unpack('<ii',f.read(8))
        a=np.frombuffer(f.read(n*k*4),dtype=np.uint32).reshape(n,k)
    return a

def topk_bruteforce(base, queries, k, mask=None):
    """top-k base ids per query by L2; if mask given, restrict base to mask ids."""
    if mask is not None:
        idx = np.nonzero(mask)[0]
        b = base[idx]
    else:
        idx = None; b = base
    bn2 = (b*b).sum(1)
    out_ids = np.empty((len(queries), k), dtype=np.uint32)
    out_d   = np.empty((len(queries), k), dtype=np.float32)
    CH = 512
    for s in range(0, len(queries), CH):
        q = queries[s:s+CH]
        d = bn2[None,:] - 2.0*(q @ b.T) + (q*q).sum(1)[:,None]
        kk = min(k, b.shape[0])
        part = np.argpartition(d, kk-1, axis=1)[:, :kk]
        rows = np.arange(len(q))[:,None]
        pd = d[rows, part]
        order = np.argsort(pd, axis=1)
        sel = part[rows, order]
        real = idx[sel] if idx is not None else sel
        out_ids[s:s+len(q), :kk] = real
        out_d[s:s+len(q), :kk]  = pd[rows, order]
        if kk < k:  # sparse label: pad
            out_ids[s:s+len(q), kk:] = 0xFFFFFFFF
            out_d[s:s+len(q), kk:] = np.inf
    return out_ids, out_d

def main():
    cmd = sys.argv[1]
    if cmd == 'genlabels':
        base_bin,N,T,mode,out_lab,out_qlab,query_bin = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8]
        base = read_vecs(base_bin)[:N]
        q = read_vecs(query_bin)
        if mode == 'diffuse':
            lab = (np.arange(N) % T).astype(np.int64)
            qlab = (np.arange(len(q)) % T).astype(np.int64)
        elif mode == 'clustered':
            # 1st principal direction (power iteration), contiguous equal blocks => spatially coherent labels
            X = base - base.mean(0)
            v = np.random.RandomState(0).randn(base.shape[1]); v/=np.linalg.norm(v)
            for _ in range(30):
                v = X.T @ (X @ v); v /= np.linalg.norm(v)
            proj = X @ v
            edges = np.quantile(proj, np.linspace(0,1,T+1))
            lab = np.clip(np.digitize(proj, edges[1:-1]), 0, T-1).astype(np.int64)
            # query label = block of its own projection (spatial predicate => high sigma_loc)
            qp = (q - base.mean(0)) @ v
            qlab = np.clip(np.digitize(qp, edges[1:-1]), 0, T-1).astype(np.int64)
        else:
            raise SystemExit('mode diffuse|clustered')
        np.savetxt(out_lab, lab, fmt='%d'); np.savetxt(out_qlab, qlab, fmt='%d')
        print(f'genlabels {mode} T={T} N={N}: base counts={np.bincount(lab).tolist()}  q counts={np.bincount(qlab).tolist()}')

    elif cmd == 'gt':
        base_bin,query_bin,labf,qlabf,N,k,out_per,out_unf,M = sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],int(sys.argv[6]),int(sys.argv[7]),sys.argv[8],sys.argv[9],int(sys.argv[10])
        base = read_vecs(base_bin)[:N]; q = read_vecs(query_bin)
        lab = read_labels(labf)[:N]; qlab = read_labels(qlabf)
        # unfiltered top-M (label-independent, reused for sigma_loc)
        uids,_ = topk_bruteforce(base, q, M)
        write_gt(out_unf, uids, np.zeros_like(uids,dtype=np.float32))
        # per-query filtered GT: top-k among base pts sharing the query's own label
        pids = np.empty((len(q), k), dtype=np.uint32)
        for L in np.unique(qlab):
            qs = np.nonzero(qlab==L)[0]
            ids,_ = topk_bruteforce(base, q[qs], k, mask=(lab==L))
            pids[qs] = ids
        write_gt(out_per, pids, np.zeros_like(pids,dtype=np.float32))
        print(f'gt: unfiltered top-{M} + per-query filtered top-{k} written ({len(q)} queries)')

    elif cmd == 'sigmaloc':
        qlabf,labf,unf,M,tag = sys.argv[2],sys.argv[3],sys.argv[4],int(sys.argv[5]),sys.argv[6]
        qlab = read_labels(qlabf); lab = read_labels(labf)
        uids = read_gt_ids(unf)[:, :M]
        inpred = (lab[uids] == qlab[:,None])            # is each unfiltered neighbour in the query's predicate
        sig = inpred.mean(1)                            # per-query local selectivity
        print(f'{tag},sigmaloc_mean={sig.mean():.5f},sigmaloc_med={np.median(sig):.5f},M={M}')

    elif cmd == 'pfrecall':
        restags,labf,qlabf,perg,k,tag = sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],int(sys.argv[6]),sys.argv[7]
        res = read_tags(restags)                        # n x K candidate tags (distance-sorted)
        lab = read_labels(labf); qlab = read_labels(qlabf)
        gt = read_gt_ids(perg)[:, :k]
        rec = np.empty(len(res))
        for i in range(len(res)):
            cand = res[i]
            cand = cand[lab[cand]==qlab[i]][:k]          # post-filter to own label, take top-k
            g = gt[i]; g = g[g!=0xFFFFFFFF]
            if len(g)==0: rec[i]=np.nan; continue
            rec[i] = len(np.intersect1d(cand, g))/len(g)
        r = np.nanmean(rec)*100
        print(f'{tag},recall@{k}={r:.3f}')

    elif cmd == 'fit':
        import csv
        recall_csv,outp = sys.argv[2], sys.argv[3]
        rows=[]
        with open(recall_csv) as f:
            for line in f:
                line=line.strip()
                if not line or line.startswith('#'): continue
                # expected: control,T,sigmaloc,L,recall
                p=line.split(',')
                try: rows.append((p[0],int(p[1]),float(p[2]),int(p[3]),float(p[4])))
                except: pass
        ctrl=np.array([r[0] for r in rows]); T=np.array([r[1] for r in rows])
        sig=np.array([r[2] for r in rows]); L=np.array([r[3] for r in rows]); rec=np.array([r[4] for r in rows])/100.0
        k=10
        x = sig*L/k
        # collapse R^2 vs a monotone logistic in log10(x)
        lx=np.log10(x)
        # simple logistic fit recall ~ 1/(1+exp(-a(lx-b)))
        from scipy.optimize import curve_fit
        def logi(z,a,b): return 1/(1+np.exp(-a*(z-b)))
        try:
            popt,_=curve_fit(logi,lx,rec,p0=[3,0],maxfev=20000)
            pred=logi(lx,*popt); ss=1-((rec-pred)**2).sum()/((rec-rec.mean())**2).sum()
        except Exception as e:
            popt=[np.nan,np.nan]; ss=np.nan
        # beta fit: recall = Pr[Binom(beta*L, sigma) >= k]
        from scipy.stats import binom
        def hbeta(beta):
            p = np.array([1-binom.cdf(k-1, max(int(round(beta*Li)),k), si) for Li,si in zip(L,sig)])
            return p
        from scipy.optimize import minimize_scalar
        def loss(beta):
            if beta<=0: return 1e9
            return ((hbeta(beta)-rec)**2).sum()
        res=minimize_scalar(loss, bounds=(0.05,5.0), method='bounded')
        beta=res.x; predb=hbeta(beta); ssb=1-((rec-predb)**2).sum()/((rec-rec.mean())**2).sum()
        # recovery law: L to hit recall>=0.95 vs 1/sigma  (per config, min L over sweep)
        print(f'COLLAPSE logistic R2={ss:.4f} (a={popt[0]:.3f}, b={popt[1]:.3f})')
        print(f'BETA fit beta={beta:.4f}  binomial R2={ssb:.4f}')
        # dump augmented csv
        with open(outp+'_fit.csv','w') as f:
            f.write('control,T,sigmaloc,L,recall,x_sigLk,pred_binom\n')
            for r,xi,pb in zip(rows,x,predb):
                f.write(f'{r[0]},{r[1]},{r[2]:.5f},{r[3]},{r[4]:.3f},{xi:.4f},{pb*100:.3f}\n')
        print(f'wrote {outp}_fit.csv')
    else:
        raise SystemExit(__doc__)

if __name__=='__main__':
    main()
