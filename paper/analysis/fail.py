import collections
from microrca.data.store import load_cases
from microrca.baselines import statistical as st
cases = load_cases()
cat = collections.Counter(); byf = collections.defaultdict(collections.Counter); ranks=[]
examples = collections.defaultdict(list)
ft_conf = collections.Counter()
for b in cases:
    d = st.diagnose(b, multimodal=True)
    pred=[c.service for c in d.ranking]; t=b.case.label.service; f=b.case.label.fault_type.value
    r = pred.index(t)+1; ranks.append((b.case.system,f,r))
    nb = {y for x,y in b.case.call_edges if x==t} | {x for x,y in b.case.call_edges if y==t}
    if r==1:
        c="hit"; ft_conf[(f, d.ranking[0].fault_type.value)]+=1
    elif pred[0] in nb: c="F1 neighbour (propagation)"
    elif r<=5: c="F2 in top-5, non-neighbour top-1"
    else: c="F3 outside top-5"
    cat[c]+=1; byf[f][c]+=1; byf[b.case.system][c]+=1
    examples[c].append((b.case.case_id,b.case.system,f,t,pred[0],r,d.ranking[0].evidence, len(b.case.services)))
print(cat); 
for k,v in byf.items(): print(k, dict(v))
print("fault-type on hits", ft_conf)
import statistics
for s in ["online-boutique","train-ticket"]:
    rs=[r for sy,f,r in ranks if sy==s]; print(s, "median rank", statistics.median(rs), "mean", round(statistics.mean(rs),1), "n", len(rs))
for k,v in examples.items():
    print("==",k); [print(e) for e in v[:4]]
# edges count
for s in ["online-boutique","train-ticket"]:
    b=[c for c in cases if c.case.system==s][0]; print(s, len(b.case.services), len(b.case.call_edges))
# fault x system counts
print(collections.Counter((c.case.system,c.case.label.fault_type.value) for c in cases))
