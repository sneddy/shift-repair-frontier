"""INLP and R-LACE at a fixed number of deletions (Corollary on the excess of a fitted subspace).

R-LACE is a batched, full-batch, fixed-budget variant: the adversarial and classifier SGD updates and the
Fantope projection follow the authors' rlace-icml implementation (Ravfogel et al., 2022); the final
iterate is rounded to a rank-k projector, without checkpoint selection or accuracy-based stopping.
INLP takes exactly k logistic nullspace steps. MP, SAL and LEACE are fitted to the same endpoints.

Usage: python inlp_rlace.py   (writes results/inlp_rlace/d*_n*.csv), then python inlp_rlace_summary.py
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import scipy
from scipy.special import ive
from scipy.stats import poisson
import sklearn
from sklearn.linear_model import LogisticRegression
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from erasers import fit_erasers

CONFIG = {
    "dimensions": [16, 32],
    "pair_count_over_dimension": [0.5, 1.0, 2.0, 4.0],
    "nullities": [1, 4],
    "repetitions_per_cell": 256,
    "theta": 1.0,
    "contrast_noise_variance": 1.0,
    "master_seed": 20260922,
    "rlace_iterations": 2000,
    "rlace_learning_rate": 0.05,
    "rlace_weight_decay": 1e-4,
    "rlace_batch": "all 2N calibration endpoints",
    "rlace_checkpoint_rule": "final iterate, rounded to fixed rank k",
    "rlace_precision": "float64",
    "inlp_classifier": "sklearn LogisticRegression, L2 penalty, C=1, fit_intercept=True, solver=lbfgs, tol=1e-9, max_iter=1000",
    "inlp_stopping": "exactly k successive nullspace steps; no validation/accuracy stopping",
    "direction": "independent uniform sphere draw per replicate",
    "comparison": "same paired calibration across all methods; no separate development data",
}


def exact_risk(d, k, kappa):
    upper = int(poisson.ppf(1 - 1e-14, kappa / 2)) + 1
    j = np.arange(upper + 1)
    return float(np.sum((d-k)*poisson.pmf(j, kappa/2)/(d+2*j)))


def posterior_coefficients(d, kappa, radius):
    a = np.sqrt(kappa) * np.asarray(radius)
    ratio = ive(d/2, a) / ive(d/2-1, a)
    q = np.where(a > 1e-6, ratio/a, 1/d)
    return q, np.maximum(0.0, 1-d*q)


def pairs(seed, n, d):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=d)
    u /= np.linalg.norm(u)
    content = rng.normal(size=(n,d))/np.sqrt(2)
    plus = content + u/2 + rng.normal(size=(n,d))/np.sqrt(2)
    minus = content - u/2 + rng.normal(size=(n,d))/np.sqrt(2)
    return plus, minus, u


def fixed_inlp(endpoints, labels, max_rank):
    d = endpoints.shape[1]
    projector = np.eye(d)
    basis = []
    outputs = {}
    diagnostics = []
    for step in range(1,max_rank+1):
        clf = LogisticRegression(C=1.0, fit_intercept=True, solver="lbfgs", tol=1e-9, max_iter=1000)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            clf.fit(endpoints @ projector, labels)
        coefficient = projector @ clf.coef_[0]
        # Reorthogonalization is numerical maintenance, not a new direction.
        if basis:
            b = np.column_stack(basis)
            coefficient -= b @ (b.T @ coefficient)
        norm = float(np.linalg.norm(coefficient))
        if norm < 1e-13:
            raise ArithmeticError(f"INLP rank exhausted at step {step}: coefficient {norm}")
        basis.append(coefficient/norm)
        b = np.column_stack(basis)
        projector = np.eye(d)-b@b.T
        outputs[step] = projector.copy()
        diagnostics.append({"step": step, "coefficient_norm": norm, "solver_iterations": int(clf.n_iter_[0]), "warnings": [str(w.message) for w in ws]})
    return outputs, diagnostics


def fantope(matrix, rank):
    """Batch version of upstream symmetric eigendecomposition + 25 bisections."""
    symmetric = (matrix+matrix.transpose(-1,-2))/2
    values, vectors = torch.linalg.eigh(symmetric)
    low = values.min(dim=-1).values-1
    high = values.max(dim=-1).values
    for _ in range(25):
        mid = (low+high)/2
        total = torch.clamp(values-mid[:,None],0,1).sum(dim=-1)
        low = torch.where(total>rank, mid, low)
        high = torch.where(total>rank, high, mid)
    clipped = torch.clamp(values-((low+high)/2)[:,None],0,1)
    return (vectors*clipped[:,None,:])@vectors.transpose(-1,-2)


def adversarial_step(x,y,p,w,b,rank,lr,decay):
    """Analytic full-batch binary gradients of the upstream alternating game."""
    m = x.shape[1]
    identity = torch.eye(x.shape[-1],dtype=x.dtype)
    p = (p+p.transpose(-1,-2))/2
    edited = x@(identity-p)
    error = torch.sigmoid(torch.einsum("bmd,bd->bm",edited,w)+b[:,None])-y
    # Gradient of negative classifier loss wrt deletion matrix P.
    xp = torch.einsum("bmd,bm->bd",x,error)/m
    gradp = xp[:,:,None]*w[:,None,:]
    p = fantope(p-lr*(gradp+decay*p),rank)
    edited = x@(identity-p)
    error = torch.sigmoid(torch.einsum("bmd,bd->bm",edited,w)+b[:,None])-y
    gradw = torch.einsum("bmd,bm->bd",edited,error)/m
    gradb = error.mean(dim=-1)
    w = w-lr*(gradw+decay*w)
    b = b-lr*(gradb+decay*b)
    return p,w,b


def rounded(p,rank):
    values,vectors=np.linalg.eigh((p+p.transpose(0,2,1))/2)
    basis=vectors[:,:,-rank:]
    return np.eye(p.shape[-1])[None,:,:]-basis@basis.transpose(0,2,1)


def batched_rlace(endpoints,labels,rank,seeds,iterations=None,checkpoint_iterations=()):
    repetitions,_,d=endpoints.shape
    x=torch.from_numpy(endpoints)
    y=torch.from_numpy(np.broadcast_to(labels,(repetitions,len(labels))).copy())
    ps=[]; ws=[]; bs=[]
    for seed in seeds:
        rng=np.random.default_rng(int(seed))
        # Same distribution as torch.nn.Linear and upstream P initialization.
        ws.append(rng.uniform(-1/np.sqrt(d),1/np.sqrt(d),d))
        bs.append(rng.uniform(-1/np.sqrt(d),1/np.sqrt(d)))
        ps.append(.1*rng.normal(size=(d,d)))
    p=torch.from_numpy(np.array(ps));w=torch.from_numpy(np.array(ws));b=torch.from_numpy(np.array(bs))
    snapshots={}
    count=CONFIG["rlace_iterations"] if iterations is None else iterations
    with torch.no_grad():
        for iteration in range(1,count+1):
            p,w,b=adversarial_step(x,y,p,w,b,rank,CONFIG["rlace_learning_rate"],CONFIG["rlace_weight_decay"])
            if iteration in checkpoint_iterations:
                snapshots[iteration]=rounded(p.numpy(),rank)
    return rounded(p.numpy(),rank),snapshots


def measure(matrix,u,w,d,k,n,q,g):
    _,singular,vt=np.linalg.svd(matrix)
    keep=singular>d*np.finfo(float).eps*singular[0]
    observed_k=int(d-keep.sum())
    if observed_k!=k:
        raise ArithmeticError(f"requested nullity {k}, observed {observed_k}")
    row=vt[keep]
    residual=float(np.square(row@u).sum())
    missed=float(np.square(row@w).sum())
    excess=float(g*missed)
    baseline=float(1-k*q-g)
    prediction=baseline+excess
    return {"residual":residual,"misalignment":missed,"predicted_excess":excess,"conditional_frontier":baseline,"posterior_prediction":prediction,"residual_minus_prediction":residual-prediction,"exact_frontier":exact_risk(d,k,n),"linear_cost":float(np.square(matrix-np.eye(d)).sum()),"observed_nullity":observed_k,"idempotence_error":float(np.linalg.norm(matrix@matrix-matrix))}


def run():
    torch.set_num_threads(1)
    output=HERE/"results"/"inlp_rlace"
    output.mkdir(parents=True,exist_ok=True)
    config=dict(CONFIG)
    config["versions"]={"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__,"torch":torch.__version__}
    config["script_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol_path=output/"protocol.json"
    if protocol_path.exists():
        previous=json.loads(protocol_path.read_text())
        if any(previous.get(key)!=value for key,value in CONFIG.items()):
            raise RuntimeError("Refusing to mix a changed configuration into persisted results.")
        expected=[output/f"d{d}_n{round(d*ratio)}.csv" for d in CONFIG["dimensions"] for ratio in CONFIG["pair_count_over_dimension"]]
        if all(path.exists() for path in expected):
            print("All declared cells already exist; no protocol or result file modified.",flush=True)
            return
    else:
        protocol_path.write_text(json.dumps(config,indent=2)+"\n")
    for d in CONFIG["dimensions"]:
        for ratio in CONFIG["pair_count_over_dimension"]:
            n=round(d*ratio)
            path=output/f"d{d}_n{n}.csv"
            if path.exists():
                print(f"Retaining completed {path.name}",flush=True);continue
            count=CONFIG["repetitions_per_cell"]
            seeds=[int(np.random.SeedSequence([CONFIG["master_seed"],d,n,j]).generate_state(1,dtype=np.uint64)[0]) for j in range(count)]
            triples=[pairs(seed,n,d) for seed in seeds]
            endpoints=np.array([np.concatenate(t[:2]) for t in triples])
            labels=np.r_[np.ones(n),np.zeros(n)]
            start=time.perf_counter()
            rl={k:batched_rlace(endpoints,labels,k,[s^0xAA88 for s in seeds])[0] for k in CONFIG["nullities"]}
            rows=[];solver=[]
            for repeat,(seed,(plus,minus,u)) in enumerate(zip(seeds,triples)):
                contrast=(plus-minus).mean(0);w=contrast/np.linalg.norm(contrast)
                q,g=posterior_coefficients(d,n,np.sqrt(n)*np.linalg.norm(contrast))
                inlp,diag=fixed_inlp(endpoints[repeat],labels,max(CONFIG["nullities"]))
                solver.append({"repeat":repeat,"seed":seed,"steps":diag})
                erasers=fit_erasers(plus,minus)
                for k in CONFIG["nullities"]:
                    rng=np.random.default_rng(seed^0xDD22)
                    random=np.column_stack((w,rng.normal(size=(d,k-1))))
                    basis=np.linalg.qr(random,mode="reduced")[0]
                    mats={"MP+random":np.eye(d)-basis@basis.T,"INLP-fixed":inlp[k],"R-LACE-fixed":rl[k][repeat]}
                    if k==1:
                        mats.update({name:fit.matrix for name,fit in erasers.items()})
                    reference=measure(mats["MP+random"],u,w,d,k,n,q,g)["residual"]
                    for name,matrix in mats.items():
                        row={"d":d,"n_pairs":n,"n_over_d":ratio,"k":k,"repeat":repeat,"seed":seed,"method":name}
                        row.update(measure(matrix,u,w,d,k,n,q,g))
                        row["paired_excess"]=row["residual"]-reference
                        row["paired_gap_error"]=row["paired_excess"]-row["predicted_excess"]
                        row["affine_cost"]=erasers[name].affine_cost if name in erasers else row["linear_cost"]
                        rows.append(row)
            with path.open("w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            (output/f"d{d}_n{n}_solver.json").write_text(json.dumps(solver,indent=2)+"\n")
            print(f"Complete d={d} N={n}: {len(rows)} fits, {time.perf_counter()-start:.1f} seconds",flush=True)


if __name__=="__main__":
    run()
