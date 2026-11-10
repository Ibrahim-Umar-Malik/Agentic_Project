import collections
from microrca.data.store import load_cases
from microrca.baselines import statistical as st
cases = load_cases()
print(collections.Counter(len(c.case.services) for c in cases if c.case.system=="train-ticket"))
for sys in ["online-boutique","train-ticket"]:
    top=collections.Counter(); miss=0
    for b in cases:
        if b.case.system!=sys: continue
        d=st.diagnose(b,multimodal=True); p=d.ranking[0].service
        if p!=b.case.label.service: top[p]+=1; miss+=1
    print(sys, miss, top.most_common(4))
# TT hit faults
for b in cases:
    d=st.diagnose(b,multimodal=True)
    if b.case.system=="train-ticket" and d.ranking[0].service==b.case.label.service: print("TT hit", b.case.case_id, b.case.label.fault_type.value)
