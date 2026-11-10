import json, random, math
def load(p): return {json.loads(l)["case_id"]: json.loads(l) for l in open(p)}
a=load("results/B0-stat-metric_cases.jsonl"); b=load("results/B0-stat-multimodal_cases.jsonl")
ids=sorted(a); 
def hit(r,k): return int(r["true_service"] in r["predicted"][:k])
for k in (1,5):
  A=[hit(a[i],k) for i in ids]; B=[hit(b[i],k) for i in ids]
  random.seed(0); diffs=[];ma=[];mb=[]
  for _ in range(10000):
    s=[random.randrange(len(ids)) for _ in ids]
    ma.append(sum(A[j] for j in s)/len(s)); mb.append(sum(B[j] for j in s)/len(s)); diffs.append(mb[-1]-ma[-1])
  q=lambda x:(round(sorted(x)[250],3),round(sorted(x)[9749],3))
  n01=sum(1 for x,y in zip(A,B) if x==0 and y==1); n10=sum(1 for x,y in zip(A,B) if x==1 and y==0)
  n=n01+n10; p=min(1,2*sum(math.comb(n,i) for i in range(0,min(n01,n10)+1))/2**n)
  print(k, "metric",q(ma),"mm",q(mb),"diff",q(diffs),"discordant b->only",n01,"a only",n10,"mcnemar p",round(p,4))
