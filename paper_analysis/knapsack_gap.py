import sys, os, pickle, json, numpy as np
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import optimal_routing as orx
R=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'city_cache')+os.sep
BUDGETS={'San_Francisco':[50,250],'Berlin':[163,816],'Singapore':[122,611]}
out={}
for city,kms in BUDGETS.items():
    G=pickle.load(open(R+city+'.pkl','rb')); edges=orx.extract_edges(G)
    val=np.array([e['weight'] for e in edges],float); cost=np.ceil(np.array([e['length_m'] for e in edges])/orx.DAS_SENSING_RANGE).astype(int)
    eff=val/np.maximum(cost,1); order=np.argsort(-eff)
    for km in kms:
        sel=orx.optimize_traffic_km(edges,km); B=int(cost[sel].sum())      # interrogator budget implied by the paper's km layout
        # greedy under interrogator budget: (i) stop at first non-fitting item (as implemented for km), (ii) skip-and-continue
        g_stop=0.0; used=0
        for i in order:
            if used+cost[i]>B: break
            used+=cost[i]; g_stop+=val[i]
        g_skip=0.0; used=0
        for i in order:
            if used+cost[i]<=B: used+=cost[i]; g_skip+=val[i]
        # exact 0-1 knapsack by DP over interrogator budget
        dp=np.zeros(B+1)
        for v,c in zip(val,cost):
            if c<=B: dp[c:]=np.maximum(dp[c:],dp[:B+1-c]+v)
        opt=float(dp.max())
        out[f'{city}_{km}km']={'B':B,'greedy_stop':g_stop,'greedy_skip':g_skip,'opt':opt,'gap_stop_pct':100*(opt-g_stop)/opt,'gap_skip_pct':100*(opt-g_skip)/opt}
        print(f"{city:<14} {km:>4} km  B={B:>5} DAS  greedy(stop)={g_stop:>10.0f}  greedy(skip)={g_skip:>10.0f}  exact={opt:>10.0f}  gap: stop {100*(opt-g_stop)/opt:.3f}%  skip {100*(opt-g_skip)/opt:.3f}%", flush=True)
json.dump(out,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'knapsack_gap.json'),'w'),indent=1)
