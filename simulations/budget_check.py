"""Fresh exact, Monte Carlo and radial-allocation check of expected budgets."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
ROOT = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(ROOT / ".matplotlib")
import csv
import hashlib
import json
import math
import sys
import time
import numpy as np
import scipy
from scipy.special import gammaln, ive
from scipy.stats import beta, chi2, ncx2, poisson
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = ROOT / "budget_results"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def mixture(kappa):
    mu=kappa/2
    lo,hi=poisson.ppf([1e-14,1-1e-14],mu).astype(int)
    j=np.arange(max(0,lo),hi+1,dtype=float)
    weights=np.exp(-mu+j*np.log(mu)-gammaln(j+1))
    return j,weights


def exact_curves(d,kappa,b):
    j,weights=mixture(kappa)
    inverse=math.fsum((weights/(d+2*j)).tolist())
    k=min(d,int(np.floor(b)))
    hard=1. if k==0 else (d-k)*inverse
    low=k;high=min(d,k+1);fraction=b-low
    low_risk=1. if low==0 else (d-low)*inverse
    high_risk=(d-high)*inverse
    naive=(1-fraction)*low_risk+fraction*high_risk
    if b==0:
        return dict(hard=hard,naive=naive,average=1.,threshold=None)
    if b==d:
        return dict(hard=hard,naive=naive,average=0.,threshold=None)
    if b==1:
        return dict(hard=hard,naive=naive,average=(d-1)*inverse,threshold=None)
    # The same radial tail supplies the weighted loss in both branches.
    lower_probability=1-b if b<1 else (b-1)/(d-1)
    threshold=float(ncx2.ppf(lower_probability,d,kappa))
    weighted=(d-1)*math.fsum((weights*chi2.sf(threshold,d+2*j)/(d+2*j)).tolist())
    average=1-b+weighted if b<1 else weighted
    return dict(hard=hard,naive=naive,average=average,threshold=threshold)


def average_summary(values,target):
    mean=float(values.mean());se=float(values.std(ddof=1)/np.sqrt(len(values)))
    return dict(mean=mean,mc_se=se,target=target,z=(mean-target)/se if se else None)


def probability_summary(mask):
    n=len(mask);count=int(np.sum(mask))
    return dict(count=count,n=n,estimate=count/n,
        ci95=[0. if count==0 else float(beta.ppf(.025,count,n-count+1)),
              1. if count==n else float(beta.ppf(.975,count+1,n-count))])


def radial_allocation(d,kappa,budgets,bins):
    # Equal probability midpoint bins; each marginal deletion costs 1/bins.
    norm2=ncx2.ppf((np.arange(bins)+.5)/bins,d,kappa)
    concentration=np.sqrt(kappa*norm2)
    gamma=ive(d/2,concentration)/(concentration*ive(d/2-1,concentration))
    parallel=1-(d-1)*gamma
    benefits=np.column_stack([parallel,np.repeat(gamma[:,None],d-1,axis=1)]).ravel()
    ordered=np.sort(benefits)[::-1]
    cumulative=np.r_[0.,np.cumsum(ordered)]
    assert np.all(parallel>1/d) and np.all(gamma<1/d)
    assert np.all(np.diff(parallel)>0) and np.all(np.diff(gamma)<0)
    rows=[]
    for b in budgets:
        units=b*bins;whole=int(np.floor(units));fraction=units-whole
        gain=cumulative[whole]+(fraction*ordered[whole] if whole<len(ordered) else 0)
        empirical_bayes=1-gain/bins
        exact=exact_curves(d,kappa,b)["average"]
        rows.append(dict(budget=b,radial_grid_risk=float(empirical_bayes),
                         exact_risk=exact,error=float(empirical_bayes-exact)))
    return dict(bins=bins,marginal_items=len(ordered),parallel_min=float(parallel.min()),
                orthogonal_max=float(gamma.max()),ordering_check="PASS",
                maximum_absolute_error=max(abs(r["error"]) for r in rows),rows=rows)


def figure(config,records,cost_distribution):
    d,kappa=config["dimension"],config["kappa"]
    grid=np.linspace(0,d,config["figure_grid_points"])
    exact=[exact_curves(d,kappa,float(b)) for b in grid]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
        "axes.labelsize":10,"axes.titlesize":10.5,"xtick.labelsize":9,"ytick.labelsize":9,
        "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.7,
        "pdf.fonttype":42,"savefig.dpi":220})
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.7),
                          gridspec_kw={"width_ratios":[1.75,1]},layout="constrained")
    ax=axes[0]
    ax.plot(grid,[r["hard"] for r in exact],color="#3e4651",drawstyle="steps-post",lw=1.6,label="Hard budget")
    ax.plot(grid,[r["naive"] for r in exact],color="#d66a24",ls="--",lw=1.7,label="Adjacent-rank mixing")
    ax.plot(grid,[r["average"] for r in exact],color="#2364aa",lw=2,label="Optimal expected budget")
    ax.errorbar([r["budget"] for r in records],[r["risk"]["mean"] for r in records],
        yerr=[1.96*r["risk"]["mc_se"] for r in records],fmt="o",color="#2364aa",
        ms=3.3,lw=.8,capsize=1.5)
    ax.set(xlabel="Allowed cost b",ylabel="Expected residual fraction",
           title=r"(a) $d=16$, $\kappa=8$",xlim=(0,d),ylim=(-.02,1.03))
    ax.set_xticks([0,4,8,12,16]);ax.grid(axis="y",alpha=.18)
    ax.legend(frameon=False,fontsize=8,loc="upper right")
    ax=axes[1];ranks=[1,d]
    probabilities=[cost_distribution[str(k)]["estimate"] for k in ranks]
    bounds=np.array([cost_distribution[str(k)]["ci95"] for k in ranks])
    ax.bar([0,1],[.8,.2],color=["#2364aa","#d66a24"],alpha=.85,width=.55)
    ax.errorbar([0,1],probabilities,yerr=np.row_stack([np.array(probabilities)-bounds[:,0],
        bounds[:,1]-probabilities]),fmt="o",color="#222222",ms=3,capsize=3,lw=.8)
    ax.text(0,.825,"80%",ha="center",fontsize=10)
    ax.text(1,.225,"20%",ha="center",fontsize=10)
    ax.set(xticks=[0,1],xticklabels=["Delete 1","Delete 16"],ylim=(0,1.02),
           ylabel="Probability",title="(b) Mean cost 4 permits cost 16")
    ax.grid(axis="y",alpha=.18);ax.set_axisbelow(True)
    for suffix in ("pdf","png"):
        fig.savefig(OUT/("expected_budget."+suffix),bbox_inches="tight",pad_inches=.04)
    plt.close(fig)


def main():
    if (OUT/"complete.json").exists():
        raise FileExistsError("Completed expected-budget run is immutable.")
    OUT.mkdir(exist_ok=True)
    config=json.loads((ROOT/"budget_config.json").read_text())
    sources={name:sha(ROOT/name) for name in ("budget_check.py","budget_config.json")}
    dump(OUT/"started.json",dict(config=config,sources=sources,python=sys.version,
                                numpy=np.__version__,scipy=scipy.__version__))
    start=time.monotonic();d,kappa,n=config["dimension"],config["kappa"],config["draws"]
    rng=np.random.default_rng(config["seed"])
    x=rng.normal(size=(n,d));x[:,0]+=np.sqrt(kappa)
    norm2=np.sum(x*x,axis=1);loss=1-x[:,0]**2/norm2
    records=[];saved=dict(norm2=norm2,projection_loss=loss)
    for b in config["budgets"]:
        exact=exact_curves(d,kappa,b)
        if b==0:cost=np.zeros(n,dtype=np.int16);risk=np.ones(n)
        elif b==d:cost=np.full(n,d,dtype=np.int16);risk=np.zeros(n)
        elif b==1:cost=np.ones(n,dtype=np.int16);risk=loss.copy()
        elif b<1:
            act=norm2>exact["threshold"];cost=act.astype(np.int16);risk=np.where(act,loss,1.)
        else:
            delete_all=norm2<=exact["threshold"]
            cost=np.where(delete_all,d,1).astype(np.int16);risk=np.where(delete_all,0.,loss)
        records.append(dict(budget=b,exact=exact,risk=average_summary(risk,exact["average"]),
                            cost=average_summary(cost,float(b))))
        if b==config["cost_distribution_budget"]:
            saved["example_cost"]=cost
            distribution={str(k):probability_summary(cost==k) for k in (1,d)}
    rotations=[]
    for index in range(config["rotational_checks"]):
        q=np.linalg.qr(rng.normal(size=(d,d)))[0]
        rotated=x@q.T;u=q[:,0];rnorm=np.sum(rotated**2,axis=1)
        rotated_loss=1-(rotated@u)**2/rnorm
        error=float(np.max(np.abs(rotated_loss-loss)))
        assert error<2e-14
        rotations.append(dict(rotation=index,maximum_risk_error=error))
    allocation=radial_allocation(d,kappa,config["budgets"],config["radial_quadrature_bins"])
    assert allocation["maximum_absolute_error"]<1e-5
    for record in records:
        assert record["exact"]["average"]<=record["exact"]["naive"]+1e-12
        assert record["exact"]["naive"]<=record["exact"]["hard"]+1e-12
    array_path=OUT/"calibration.npz";np.savez_compressed(array_path,**saved)
    dump(OUT/"results.json",dict(config=config,records=records,cost_distribution=distribution,
                                radial_allocation=allocation,rotational_checks=rotations))
    with (OUT/"results.csv").open("w",newline="") as handle:
        keys=["budget","hard_risk","naive_risk","expected_budget_risk","mc_risk","risk_mc_se","mc_cost","cost_mc_se"]
        writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader()
        for r in records:
            writer.writerow(dict(budget=r["budget"],hard_risk=r["exact"]["hard"],naive_risk=r["exact"]["naive"],
                expected_budget_risk=r["exact"]["average"],mc_risk=r["risk"]["mean"],
                risk_mc_se=r["risk"]["mc_se"],mc_cost=r["cost"]["mean"],cost_mc_se=r["cost"]["mc_se"]))
    figure(config,records,distribution)
    risk_z=max(abs(r["risk"]["z"]) for r in records if r["risk"]["z"] is not None)
    cost_z=max(abs(r["cost"]["z"]) for r in records if r["cost"]["z"] is not None)
    dump(OUT/"complete.json",dict(status="PASS",sources=sources,independent_calibration_draws=n,
        budgets_evaluated=len(records),max_abs_risk_mc_z=risk_z,max_abs_cost_mc_z=cost_z,
        radial_allocation_max_error=allocation["maximum_absolute_error"],seconds=time.monotonic()-start,
        files={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!="complete.json"}))
    print(json.dumps(dict(status="PASS",draws=n,risk_max_abs_z=risk_z,cost_max_abs_z=cost_z,
                          radial_allocation_max_error=allocation["maximum_absolute_error"],
                          budget4=next(r for r in records if r["budget"]==4)),indent=2))


if __name__=="__main__":main()
