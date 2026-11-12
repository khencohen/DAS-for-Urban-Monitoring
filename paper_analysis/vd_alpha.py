import os, sys, glob, json, numpy as np, networkx as nx
sys.path.insert(0, os.path.dirname(__file__)); from common import load_graph, CACHE
rng=np.random.default_rng(0); DMAX=10; NS=300; P=0.9; res={}
for p in sorted(glob.glob(CACHE+'/*.pkl')):
    name=os.path.basename(p)[:-4]; G=load_graph(name)
    G=G.subgraph(max(nx.connected_components(G),key=len)).copy()
    nodes=list(G.nodes()); k=2*G.number_of_edges()/G.number_of_nodes()
    Ve=np.zeros(DMAX+2); Vn=np.zeros(DMAX+2)
    for si in rng.choice(len(nodes),NS,replace=False):
        dist=nx.single_source_shortest_path_length(G,nodes[si],cutoff=DMAX)
        for n,dn in dist.items():
            Vn[dn]+=1
            for m in G[n]:
                dm=dist.get(m)
                if dm is None or dm>dn or (dm==dn and n<m):   # count edge once, at hop level of nearer endpoint +1
                    Ve[dn+1]+=1
    Ve=np.cumsum(Ve)/NS; Vn=np.cumsum(Vn)/NS         # cumulative within d
    d=np.arange(2,9); alpha_e=float(np.sum(Ve[d]*d**2)/np.sum(d**4)); alpha_n=float(np.sum(Vn[d]*d**2)/np.sum(d**4))
    # coverage c needed for P=0.9 at d=10 (EEW-type), and d at c=0.10, with measured alpha_e
    d_at_c10=float(np.sqrt(np.log(1-P)/np.log(1-0.10)/alpha_e))
    res[name]={'k':k,'alpha_edges':alpha_e,'alpha_nodes':alpha_n,'k_over_2':k/2,'Ve':Ve[:DMAX+1].tolist(),'Vn':Vn[:DMAX+1].tolist(),'d_at_c0.10':d_at_c10}
    print(f"{name:<14} <k>={k:.2f}  alpha_edges={alpha_e:5.2f}  alpha_nodes={alpha_n:5.2f}  <k>/2={k/2:.2f}  Ve(5)/25={Ve[5]/25:.2f}  d(c=10%)={d_at_c10:.2f}", flush=True)
a=np.array([r['alpha_edges'] for r in res.values()])
print(f"\nalpha_edges: mean={a.mean():.2f} sd={a.std(ddof=1):.2f} min={a.min():.2f} max={a.max():.2f}")
dd=np.array([r['d_at_c0.10'] for r in res.values()]); print(f"d(c=10%): mean={dd.mean():.2f} range {dd.min():.2f}-{dd.max():.2f}  (model alpha=2 -> {np.sqrt(np.log(0.1)/np.log(0.9)/2):.2f})")
json.dump(res,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'vd_alpha.json'),'w'),indent=1)
