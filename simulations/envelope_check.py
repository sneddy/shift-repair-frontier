"""Exact component envelope, independent integrals, and existing-array replay."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[key]="1"
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"]=str(ROOT/".matplotlib")
import csv
import hashlib
import json
import math
import sys
import time
import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import gammaln
from scipy.stats import beta, chi2, f, ncf, ncx2, norm, poisson
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT=ROOT/"envelope_results"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,data):Path(path).write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+"\n")


def envelope(d,alpha,epsilon,nu=None):
    # Markov gives B_j <= (d-1)/(epsilon*(d+2j)).
    cap=int(np.ceil(max(0,((d-1)/(alpha*epsilon)-d)/2)))+2
    js=np.arange(cap+1);B=beta.sf(epsilon,(d-1)/2,(1+2*js)/2)
    cut=int(np.flatnonzero(B<=alpha)[0])
    active=np.arange(cut);values=B[:cut]
    # Strict mathematical monotonicity can round to equal tails near one.
    differences=np.diff(B[:cut+1])
    assert np.all(differences<=0)
    if cut:
        assert np.any(differences<0)
    if nu is None:
        null=float(chi2.isf(alpha,d));old=null/epsilon
        orthogonal=max(null,float(chi2.isf(alpha,d-1))/epsilon)
        at_null=values*chi2.sf(null,d+2*active)
        needed=at_null>alpha
        candidates=np.full(cut,null)
        candidates[needed]=chi2.isf(alpha/values[needed],d+2*active[needed])
    else:
        null=float(f.isf(alpha,d,nu));old=null/epsilon
        orthogonal=max(null,(d-1)/d*float(f.isf(alpha,d-1,nu))/epsilon)
        at_null=values*f.sf(d*null/(d+2*active),d+2*active,nu)
        needed=at_null>alpha
        candidates=np.full(cut,null)
        candidates[needed]=(d+2*active[needed])/d*f.isf(alpha/values[needed],d+2*active[needed],nu)
    threshold=float(max(null,float(candidates.max()) if cut else null))
    if nu is None:
        component=values*chi2.sf(threshold,d+2*active);null_action=float(chi2.sf(threshold,d))
    else:
        component=values*f.sf(d*threshold/(d+2*active),d+2*active,nu);null_action=float(f.sf(threshold,d,nu))
    max_component=float(component.max()) if cut else 0.
    assert max_component<=alpha+2e-13 and null_action<=alpha+2e-13
    assert threshold<=orthogonal+2e-12 and orthogonal<old and B[cut]<=alpha
    critical=int(np.argmax(candidates)) if cut and threshold>null+1e-12 else None
    return dict(d=d,nu=nu,threshold=threshold,ball_threshold=old,orthogonal_threshold=orthogonal,null_threshold=null,
        threshold_ratio=threshold/old,first_unconstraining_j=cut,markov_index_cap=cap,
        beta_at_cutoff=float(B[cut]),critical_j=critical,
        maximum_constraining_component=max_component,null_action=null_action,
        finite_component_check="PASS",tail_component_check="PASS",strict_ball_improvement=True)


def exact_joint(d,kappa,threshold,epsilon,nu=None):
    if kappa==0:
        j=np.array([0.]);weights=np.array([1.])
    else:
        mu=kappa/2;lo,hi=poisson.ppf([1e-14,1-1e-14],mu).astype(int)
        j=np.arange(max(0,lo),hi+1,dtype=float)
        weights=np.exp(-mu+j*np.log(mu)-gammaln(j+1))
    degrees=d+2*j
    tail=chi2.sf(threshold,degrees) if nu is None else f.sf(d*threshold/degrees,degrees,nu)
    bad=beta.sf(epsilon,(d-1)/2,(1+2*j)/2)
    return math.fsum((weights*tail*bad).tolist())


def action_power(d,kappa,threshold,nu=None):
    if nu is None:return float(ncx2.sf(threshold,d,kappa))
    return float(ncf.sf(threshold,d,nu,kappa) if kappa else f.sf(threshold,d,nu))


def independent_known(d,kappa,threshold,epsilon):
    mu=np.sqrt(kappa);lo,hi=mu-10,mu+10
    corner=np.sqrt(threshold*(1-epsilon))
    cuts=[lo]+sorted(z for z in (-corner,0.,corner) if lo<z<hi)+[hi]
    fn=lambda z:norm.pdf(z-mu)*chi2.sf(max(threshold-z*z,epsilon*z*z/(1-epsilon)),d-1)
    values=[quad(fn,a,b,epsabs=2e-12,epsrel=2e-10,limit=150) for a,b in zip(cuts[:-1],cuts[1:])]
    return dict(value=float(sum(v[0] for v in values)),reported_quad_error=float(sum(v[1] for v in values)),
                omitted_normal_mass=float(2*norm.sf(10)))


def independent_unknown(d,nu,kappa,threshold,epsilon):
    mu=np.sqrt(kappa)
    def integrand(z):
        lower=epsilon*z*z/(1-epsilon)
        def inner(v):
            return chi2.pdf(v,d-1)*chi2.cdf(nu*(z*z+v)/(d*threshold),nu)
        value,_=quad(inner,lower,np.inf,epsabs=3e-11,epsrel=3e-9,limit=150)
        return norm.pdf(z-mu)*value
    value,error=quad(integrand,mu-10,mu+10,epsabs=2e-9,epsrel=2e-8,limit=150)
    return dict(value=float(value),reported_outer_quad_error=float(error),
                inner_epsabs=3e-11,omitted_normal_mass=float(2*norm.sf(10)))


def probability(mask):
    n=len(mask);count=int(mask.sum())
    return dict(count=count,n=n,estimate=count/n,
        ci95=[0. if count==0 else float(beta.ppf(.025,count,n-count+1)),
              1. if count==n else float(beta.ppf(.975,count+1,n-count))])


def summarize_arrays(d,kappa,norm2,loss,thresholds,epsilon,nu=None,w=None):
    statistic=norm2 if nu is None else (norm2/d)/(w/nu)
    result={}
    for name,threshold in thresholds.items():
        act=statistic>threshold
        row=dict(threshold=threshold,action=probability(act),
                 exact_action=action_power(d,kappa,threshold,nu))
        if kappa>0:
            row.update(joint_bad=probability(act&(loss>epsilon)),
                       exact_joint_bad=exact_joint(d,kappa,threshold,epsilon,nu),
                       loss_including_abstention=float(np.where(act,loss,1.).mean()))
        else:
            row.update(joint_bad=None,exact_joint_bad=None,loss_including_abstention=None)
        result[name]=row
    assert result["envelope"]["action"]["count"]>=result["ball"]["action"]["count"]
    return result


def replay(config,threshold_records):
    primary=ROOT/"results"
    completed=json.loads((primary/"complete.json").read_text())
    threshold_map={(r["d"],r["nu"]):r for r in threshold_records}
    groups={};inputs={}
    for receipt in completed["arrays"]:
        if receipt["kind"] not in {"isotropic","unknown_scale"}:continue
        path=primary/receipt["file"]
        assert sha(path)==receipt["sha256"]
        inputs[str(path.relative_to(ROOT))]=receipt["sha256"]
        with np.load(path,allow_pickle=False) as z:
            d=receipt["d"];kappa=receipt["kappa"]
            if receipt["kind"]=="isotropic":
                key=("known",d,None,kappa)
                groups.setdefault(key,[]).append(dict(norm2=z["norm2"].copy(),loss=z["losses"][:,0].copy()))
            else:
                N=receipt["N"];key=("unknown",d,N,kappa)
                groups.setdefault(key,[]).append(dict(norm2=z["norm2"].copy(),loss=z["loss"].copy(),w=z["w"].copy()))
                if d==4 and N==2:
                    groups.setdefault(("known",d,None,kappa),[]).append(dict(norm2=z["norm2"].copy(),loss=z["loss"].copy()))
    rows=[]
    for (kind,d,N,kappa),arrays in sorted(groups.items(),key=lambda p:str(p[0])):
        nu=d*(N-1) if N is not None else None
        threshold=threshold_map[(d,nu)]
        norm2=np.concatenate([z["norm2"] for z in arrays]);loss=np.concatenate([z["loss"] for z in arrays])
        w=np.concatenate([z["w"] for z in arrays]) if nu is not None else None
        policies=summarize_arrays(d,kappa,norm2,loss,dict(ball=threshold["ball_threshold"],
                     orthogonal=threshold["orthogonal_threshold"],
                     envelope=threshold["threshold"]),config["epsilon"],nu,w)
        rows.append(dict(kind=kind,d=d,N=N,nu=nu,kappa=kappa,ratio=kappa/d,n=len(norm2),policies=policies))
    return rows,inputs


def figures(config,thresholds,rows):
    threshold_map={(r["d"],r["nu"]):r for r in thresholds}
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.labelsize":10,
        "axes.titlesize":11,"xtick.labelsize":9,"ytick.labelsize":9,
        "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.7,
        "pdf.fonttype":42,"savefig.dpi":220})
    xs=np.linspace(0,config["figure_max_kappa_over_d"],config["figure_points"])
    fig,axes=plt.subplots(1,3,figsize=(7.1,2.6),layout="constrained",sharey=True)
    for ax,d in zip(axes,config["figure_dimensions"]):
        t=threshold_map[(d,None)]
        for name,key,color,marker,label,style in [("ball","ball_threshold","#d66a24","s","Confidence ball","-"),
                                           ("orthogonal","orthogonal_threshold","#596a59","^","Orthogonal energy","--"),
                                           ("envelope","threshold","#2364aa","o","Joint-error envelope","-")]:
            ax.plot(xs,[action_power(d,d*x,t[key]) for x in xs],color=color,lw=1.9,label=label,ls=style)
            selected=sorted([r for r in rows if r["kind"]=="known" and r["d"]==d and r["ratio"]<=xs[-1]],key=lambda r:r["ratio"])
            values=np.array([r["policies"][name]["action"]["estimate"] for r in selected])
            intervals=np.array([r["policies"][name]["action"]["ci95"] for r in selected])
            ax.errorbar([r["ratio"] for r in selected],values,
                yerr=np.row_stack([values-intervals[:,0],intervals[:,1]-values]),
                fmt=marker,color=color,ms=3,lw=.8,capsize=1.5)
        ax.set(title=f"d = {d}",xlabel=r"Calibration strength $\kappa/d$",xlim=(0,xs[-1]),ylim=(-.02,1.02))
        ax.set_xticks([0,1,2,3,4]);ax.grid(axis="y",alpha=.18)
    axes[0].set_ylabel("Action probability")
    axes[0].legend(frameon=False,fontsize=7.8,loc="upper left")
    for suffix in ("pdf","png"):fig.savefig(OUT/("gate_comparison."+suffix),bbox_inches="tight",pad_inches=.04)
    plt.close(fig)
    # Optional companion separates the guarantee's joint event from action power.
    fig,axes=plt.subplots(1,3,figsize=(7.1,2.4),layout="constrained",sharey=True)
    for ax,d in zip(axes,config["figure_dimensions"]):
        t=threshold_map[(d,None)]
        for key,color,label,style in [("ball_threshold","#d66a24","Confidence ball","-"),
                ("orthogonal_threshold","#596a59","Orthogonal energy","--"),("threshold","#2364aa","Envelope","-")]:
            ax.plot(xs,[exact_joint(d,d*x,t[key],config["epsilon"]) for x in xs],color=color,lw=1.8,label=label,ls=style)
        ax.axhline(config["alpha"],color="#3e4651",ls="--",lw=1,label=r"$\alpha=0.05$")
        ax.set(title=f"d = {d}",xlabel=r"Calibration strength $\kappa/d$",xlim=(0,xs[-1]),ylim=(0,.055))
        ax.grid(axis="y",alpha=.18)
    axes[0].set_ylabel("Joint bad-action probability")
    axes[0].legend(frameon=False,fontsize=7.7,loc="center right")
    for suffix in ("pdf","png"):fig.savefig(OUT/("gate_safety."+suffix),bbox_inches="tight",pad_inches=.04)
    plt.close(fig)


def main():
    if (OUT/"complete.json").exists():raise FileExistsError("Completed envelope analysis is immutable.")
    OUT.mkdir(exist_ok=True)
    config=json.loads((ROOT/"envelope_config.json").read_text())
    sources={name:sha(ROOT/name) for name in ("envelope_check.py","envelope_config.json")}
    dump(OUT/"started.json",dict(config=config,sources=sources,new_gaussian_draws=0,
         research_stage="Post-run analytic extension on frozen existing arrays",
         python=sys.version,numpy=np.__version__,scipy=scipy.__version__))
    start=time.monotonic();thresholds=[]
    for d in config["dimensions"]:
        thresholds.append(envelope(d,config["alpha"],config["epsilon"]))
        for N in config["unknown_scale_unit_counts"]:
            row=envelope(d,config["alpha"],config["epsilon"],d*(N-1));row["N"]=N
            thresholds.append(row)
    dump(OUT/"thresholds.json",thresholds)
    print("Finite component thresholds all pass.",flush=True)
    rows,inputs=replay(config,thresholds)
    dump(OUT/"replay.json",rows)
    tmap={(r["d"],r["nu"]):r for r in thresholds};integration=[]
    for d,kappa in config["known_quadrature_cases"]:
        for name in ("threshold","ball_threshold"):
            t=tmap[(d,None)][name]
            checked=independent_known(d,kappa,t,config["epsilon"])
            exact=exact_joint(d,kappa,t,config["epsilon"])
            checked.update(d=d,kappa=kappa,nu=None,gate=name,mixture=exact,
                           absolute_error=abs(checked["value"]-exact))
            assert checked["absolute_error"]<2e-9
            integration.append(checked)
    print("Independent Gaussian/orthogonal-norm integrals pass.",flush=True)
    for d,N,kappa in config["unknown_quadrature_cases"]:
        nu=d*(N-1);t=tmap[(d,nu)]["threshold"]
        checked=independent_unknown(d,nu,kappa,t,config["epsilon"])
        exact=exact_joint(d,kappa,t,config["epsilon"],nu)
        checked.update(d=d,kappa=kappa,N=N,nu=nu,gate="threshold",mixture=exact,
                       absolute_error=abs(checked["value"]-exact))
        assert checked["absolute_error"]<2e-8
        integration.append(checked)
    dump(OUT/"independent_integrals.json",integration)
    safety=[]
    ratios=np.geomspace(config["safety_kappa_over_d_min"],config["safety_kappa_over_d_max"],config["safety_grid_points"])
    for record in thresholds:
        d,nu=record["d"],record["nu"]
        values=np.array([exact_joint(d,d*r,record["threshold"],config["epsilon"],nu) for r in ratios])
        largest=int(values.argmax());assert values[largest]<config["alpha"]+1e-12
        safety.append(dict(d=d,nu=nu,max_grid_joint_bad=float(values[largest]),
                           kappa_at_max=float(d*ratios[largest]),grid_points=len(ratios)))
    dump(OUT/"safety_grid.json",safety)
    with (OUT/"thresholds.csv").open("w",newline="") as handle:
        fields=["d","nu","ball_threshold","threshold","threshold_ratio","critical_j","first_unconstraining_j",
                "maximum_constraining_component","null_action"]
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(thresholds)
    tex=[]
    for r in thresholds:
        nu="--" if r["nu"] is None else str(r["nu"])
        tex.append(f"{r['d']} & {nu} & {r['ball_threshold']:.4f} & {r['threshold']:.4f} & {r['first_unconstraining_j']} & {r['maximum_constraining_component']:.4f}"+ " "+chr(92)*2)
    (OUT/"envelope_threshold_rows.tex").write_text("\n".join(tex)+"\n")
    figures(config,thresholds,rows)
    dump(OUT/"complete.json",dict(status="PASS",sources=sources,new_gaussian_draws=0,
        input_array_count=len(inputs),input_arrays=inputs,threshold_rows=len(thresholds),replay_rows=len(rows),
        distinct_original_draws=380928,
        independent_integration_checks=len(integration),
        maximum_independent_integration_error=max(r["absolute_error"] for r in integration),
        maximum_finite_component=max(r["maximum_constraining_component"] for r in thresholds),
        seconds=time.monotonic()-start,
        files={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!="complete.json"}))
    print(json.dumps(dict(status="PASS",known_thresholds=[r for r in thresholds if r["nu"] is None],
        max_integration_error=max(r["absolute_error"] for r in integration),
        new_gaussian_draws=0),indent=2),flush=True)


if __name__=="__main__":main()
