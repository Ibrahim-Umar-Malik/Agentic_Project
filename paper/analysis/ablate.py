import time, collections, json
from microrca.data.store import load_cases
from microrca.baselines import statistical as st
from microrca.eval.metrics import ac_at_k, avg_at_k
from microrca.contracts import EvalRecord, Usage

cases = load_cases()
def rank_with(bundle, signals):
    services = bundle.case.services
    fused = {s: 0.0 for s in services}
    for name in signals:
        scores = {s: st._signal(bundle, s, name) for s in services}
        for r, s in enumerate(sorted(services, key=scores.get, reverse=True), start=1):
            fused[s] += 1.0 / r
    return sorted(services, key=fused.get, reverse=True)

def evaluate(signals):
    recs=[]; t=time.time()
    for b in cases:
        p = rank_with(b, signals)
        recs.append(EvalRecord(case_id=b.case.case_id, system=b.case.system, method="x", true_service=b.case.label.service,
            true_fault=b.case.label.fault_type.value, predicted=p, predicted_fault="unknown", usage=Usage()))
    ob=[r for r in recs if r.system=="online-boutique"]; tt=[r for r in recs if r.system=="train-ticket"]
    return dict(AC1=round(ac_at_k(recs,1),3),AC3=round(ac_at_k(recs,3),3),AC5=round(ac_at_k(recs,5),3),Avg5=round(avg_at_k(recs,5),3),
                OB1=round(ac_at_k(ob,1),3),TT1=round(ac_at_k(tt,1),3), ms=round((time.time()-t)/len(cases)*1000,2))
F = st.FUSED_SIGNALS
print("full", F, evaluate(F))
for s in F:
    print("-", s, evaluate([x for x in F if x!=s]))
for s in F:
    print("only", s, evaluate([s]))
print("metrics-only fused (cpu,mem,lat)", evaluate(["cpu","memory","server_latency_p90"]))
