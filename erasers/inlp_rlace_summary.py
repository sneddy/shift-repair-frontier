"""Summarize the INLP / R-LACE runs into results/inlp_rlace/summary.{csv,json}."""
import csv
import json
from pathlib import Path
import numpy as np
from inlp_rlace import CONFIG, exact_risk

HERE=Path(__file__).resolve().parent
OUT=HERE/"results"/"inlp_rlace"


def stats(values):
    v=np.array(values,dtype=float)
    return {"mean":float(v.mean()),"se":float(v.std(ddof=1)/np.sqrt(len(v))),"median":float(np.median(v)),"q25":float(np.quantile(v,.25)),"q75":float(np.quantile(v,.75)),"min":float(v.min()),"max":float(v.max())}


def main():
    raw=[]
    for path in sorted(OUT.glob("d*_n*.csv")):
        with path.open() as f:
            for r in csv.DictReader(f):
                raw.append({key:value if key=="method" else int(value) if key in ("d","n_pairs","k","repeat","seed","observed_nullity") else float(value) for key,value in r.items()})
    assert len(raw)==len(CONFIG["dimensions"])*len(CONFIG["pair_count_over_dimension"])*CONFIG["repetitions_per_cell"]*9,len(raw)
    groups=sorted({(r["d"],r["n_pairs"],r["k"],r["method"]) for r in raw})
    cells=[]
    metrics=["residual","predicted_excess","posterior_prediction","residual_minus_prediction","paired_excess","paired_gap_error","misalignment","linear_cost","affine_cost"]
    for d,n,k,name in groups:
        rows=[r for r in raw if (r["d"],r["n_pairs"],r["k"],r["method"])==(d,n,k,name)]
        assert len(rows)==CONFIG["repetitions_per_cell"]
        item={"d":d,"n_pairs":n,"n_over_d":n/d,"k":k,"method":name,"repetitions":len(rows),"exact_frontier":exact_risk(d,k,n)}
        for metric in metrics:
            for stat,value in stats([r[metric] for r in rows]).items():
                item[f"{metric}_{stat}"]=value
        item["unconditional_prediction"]=item["exact_frontier"]+item["predicted_excess_mean"]
        for metric in ["residual_minus_prediction","paired_gap_error"]:
            item[f"{metric}_z"]=item[f"{metric}_mean"]/item[f"{metric}_se"] if item[f"{metric}_se"]>1e-12 else 0.
        cells.append(item)
    with (OUT/"summary.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(cells[0]));writer.writeheader();writer.writerows(cells)
    independent=[c for c in cells if c["method"] in ("INLP-fixed","R-LACE-fixed")]
    solver=[]
    for path in sorted(OUT.glob("*_solver.json")):
        solver.extend(json.loads(path.read_text()))
    report={"config":CONFIG,"total_method_fits":len(raw),"total_distinct_calibrations":len(raw)//9,"cells":cells,"max_idempotence_error":max(r["idempotence_error"] for r in raw),"identity_rmse_across_method_cells":float(np.sqrt(np.mean([c["residual_minus_prediction_mean"]**2 for c in independent]))),"paired_excess_prediction_rmse":float(np.sqrt(np.mean([c["paired_gap_error_mean"]**2 for c in independent]))),"max_abs_identity_z":max(abs(c["residual_minus_prediction_z"]) for c in independent),"max_abs_paired_gap_z":max(abs(c["paired_gap_error_z"]) for c in independent),"inlp_min_coefficient_norm":min(s["coefficient_norm"] for r in solver for s in r["steps"]),"inlp_max_solver_iterations":max(s["solver_iterations"] for r in solver for s in r["steps"]),"inlp_solver_warning_count":sum(len(s["warnings"]) for r in solver for s in r["steps"]),"no_filtered_fits":True}
    (OUT/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("config","cells")},indent=2))



if __name__=="__main__":main()
