"""Monte Carlo checks of the frontier and of the action rules (known and unknown scale).

Usage: python run.py   (writes results/; refuses to overwrite a complete run)
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
import scipy
from scipy.stats import beta, chi2, f, ncf, ncx2
from model import exact_risk, exact_operating, mean_summary, policy_summary, probability, poisson_terms

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def save(stem, arrays, meta, receipts):
    path = OUT/'arrays'/(stem+'.npz')
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(path, **arrays)
    receipts.append(dict(**meta, file=str(path.relative_to(OUT)), sha256=sha(path)))


def run_isotropic(config, alpha, epsilon, receipts):
    rows = []
    for di, d in enumerate(config['dimensions']):
        for ri, ratio in enumerate(config['lambda_over_d']):
            kappa = d*ratio
            ranks = sorted({1, d//4, d//2, d})
            collected = []
            for seed in config['seeds']:
                rng = np.random.default_rng(np.random.SeedSequence([seed, di, ri, 11]))
                n = config['draws_per_seed_cell']
                x = rng.normal(size=(n,d))
                x[:,0] += np.sqrt(kappa)
                norm2 = np.sum(x*x, axis=1)
                loss = 1-x[:,0]**2/norm2
                losses = []
                for k in ranks:
                    losses.append(loss if k==1 else np.zeros(n) if k==d else
                                  loss*rng.beta((d-k)/2, (k-1)/2, size=n))
                arrays = dict(norm2=norm2, losses=np.column_stack(losses))
                save(f'iso_d{d}_ratio{ratio:g}_seed{seed}', arrays,
                     dict(kind='isotropic', d=d, kappa=kappa, seed=seed, n=n), receipts)
                collected.append(arrays)
            norm2 = np.concatenate([a['norm2'] for a in collected])
            losses = np.concatenate([a['losses'] for a in collected])
            q = float(chi2.ppf(1-alpha,d))
            row = dict(d=d, kappa=kappa, ratio=ratio, n=len(norm2), frontier=[], policies={})
            for index, k in enumerate(ranks):
                row['frontier'].append(dict(k=k, exact=exact_risk(d,kappa,k) if kappa else None,
                    empirical=mean_summary(losses[:,index]) if kappa else None,
                    directional_limit=mean_summary(losses[:,index]) if not kappa else None))
            for name, threshold in [('detect',q), ('action',q/epsilon)]:
                row['policies'][name] = policy_summary(norm2>threshold, losses[:,0] if kappa else None)
                row['policies'][name]['exact_action'] = float(ncx2.sf(threshold,d,kappa))
                row['policies'][name]['exact_operating'] = exact_operating(d,kappa,threshold,epsilon) if kappa else None
            rows.append(row)
    return rows


def unknown_operating(d,nu,kappa,threshold,epsilon):
    j,w = poisson_terms(d,kappa)
    degrees = d+2*j
    tails = f.sf(d*threshold/degrees,degrees,nu)
    a,b = (d-1)/2,(1+2*j)/2
    p = float(w@tails)
    weighted = float((w*tails)@(a/(a+b)))
    return dict(action=p,joint_bad=float((w*tails)@beta.sf(epsilon,a,b)),
                loss_including_abstention=1-p+weighted,
                loss_on_action=weighted/p if p else None)


def run_unknown(config,alpha,epsilon,receipts):
    rows = []
    for di,d in enumerate(config['dimensions']):
        for ni,N in enumerate(config['pair_counts']):
            nu = d*(N-1)
            q = float(f.ppf(1-alpha,d,nu))
            for ri,ratio in enumerate(config['lambda_over_d']):
                kappa=d*ratio
                n=config['draws_per_cell']
                rng=np.random.default_rng(np.random.SeedSequence([config['seed'],di,ni,ri,13]))
                noise=rng.normal(size=(n,d))
                x=noise.copy();x[:,0]+=np.sqrt(kappa)
                w=rng.chisquare(nu,size=n)
                norm2=np.sum(x*x,axis=1)
                fstat=(norm2/d)/(w/nu)
                loss=1-x[:,0]**2/norm2
                arrays=dict(norm2=norm2,w=w,loss=loss,
                    known_detect=norm2>chi2.ppf(1-alpha,d),
                    known_action=norm2>chi2.ppf(1-alpha,d)/epsilon,
                    unknown_detect=fstat>q,unknown_action=fstat>q/epsilon,
                    covered=np.sum(noise**2,axis=1)<=d*q*w/nu)
                save(f'unknown_d{d}_N{N}_ratio{ratio:g}',arrays,
                     dict(kind='unknown_scale',d=d,N=N,kappa=kappa,seed=config['seed'],n=n),receipts)
                row=dict(d=d,N=N,nu=nu,kappa=kappa,ratio=ratio,n=n,q=q,
                         coverage=probability(arrays['covered']),policies={})
                for name in ('known_detect','known_action','unknown_detect','unknown_action'):
                    multiplier=epsilon if name.endswith('action') else 1.
                    if name.startswith('unknown'):
                        threshold=q/multiplier
                        exact=float(ncf.sf(threshold,d,nu,kappa) if kappa else f.sf(threshold,d,nu))
                        operating=unknown_operating(d,nu,kappa,threshold,epsilon) if kappa else None
                    else:
                        threshold=chi2.ppf(1-alpha,d)/multiplier
                        exact=float(ncx2.sf(threshold,d,kappa))
                        operating=exact_operating(d,kappa,threshold,epsilon) if kappa else None
                    row['policies'][name]=policy_summary(arrays[name],loss if kappa else None)
                    row['policies'][name].update(exact_action=exact,exact_operating=operating)
                rows.append(row)
    return rows


def main():
    if (OUT/'complete.json').exists():
        raise FileExistsError('Completed results are immutable; use a fresh directory for a new run.')
    (OUT/'arrays').mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    config=json.loads((ROOT/'config.json').read_text())
    sources={name:sha(ROOT/name) for name in ('config.json','model.py','run.py')}
    environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
                     numpy=np.__version__,scipy=scipy.__version__)
    dump(OUT/'started.json',dict(config=config,sources=sources,environment=environment))
    receipts=[]
    records={}
    for name,runner in [('isotropic',run_isotropic),('unknown_scale',run_unknown)]:
        records[name]=runner(config[name],config['alpha'],config['epsilon'],receipts)
        dump(OUT/(name+'.json'),records[name])
        print(name,len(records[name]),'pooled cells complete',flush=True)
    counts={name:dict(generated_seed_cells=sum(r['kind']==name for r in receipts),
                 draws=sum(r['n'] for r in receipts if r['kind']==name),pooled_parameter_cells=len(rows))
            for name,rows in records.items()}
    dump(OUT/'complete.json',dict(status='COMPLETE',sources=sources,environment=environment,
         counts=counts,arrays=receipts,seconds=time.monotonic()-started,
         result_hashes={name+'.json':sha(OUT/(name+'.json')) for name in records}))
    print(json.dumps(counts),flush=True)


if __name__=='__main__': main()
