import os, sys, json, pickle, numpy as np, networkx as nx
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
sys.path.insert(0, os.path.dirname(__file__)); from common import load_graph
OUT = os.path.dirname(os.path.abspath(__file__))
C_STAR = 0.516; SEED = 42
CITIES = ['Tel_Aviv','Barcelona','Manhattan','Berlin','Paris','Amsterdam','Rome','Buenos_Aires','San_Francisco','Tokyo']
NAMES = {'Tel_Aviv':'Tel Aviv','Buenos_Aires':'Buenos Aires','San_Francisco':'San Francisco','Tokyo':'Shibuya (Tokyo)'}

def gabriel_graph(n, rng):
    pts = rng.random((n, 2)); tri = Delaunay(pts); cand = set()
    for s in tri.simplices:
        for a, b in ((s[0],s[1]),(s[1],s[2]),(s[2],s[0])): cand.add((min(a,b),max(a,b)))
    adj = {i:set() for i in range(n)}
    for u,v in cand: adj[u].add(v); adj[v].add(u)
    E=[]
    for u,v in cand:
        c=(pts[u]+pts[v])/2; r2=np.sum((pts[u]-pts[v])**2)/4
        if not any(np.sum((pts[w]-c)**2) < r2-1e-9 for w in adj[u]&adj[v]): E.append((u,v))
    G=nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(E); return G

def percolation_curve(G, n_runs, n_steps, rng):
    edges=list(G.edges()); m,n=len(edges),G.number_of_nodes()
    cs=np.linspace(0,1,n_steps); gcc=np.zeros(n_steps)
    for _ in range(n_runs):
        p=rng.random(m)
        for i,c in enumerate(cs):
            H=nx.Graph(); H.add_nodes_from(G.nodes()); H.add_edges_from(edges[j] for j in np.where(p<c)[0])
            if H.number_of_edges(): gcc[i]+=len(max(nx.connected_components(H),key=len))/n
    return cs, gcc/n_runs

def threshold_at(cs,gcc,t=0.5):
    for i in range(1,len(cs)):
        if gcc[i]>=t>gcc[i-1]:
            s=(gcc[i]-gcc[i-1])/(cs[i]-cs[i-1]); return cs[i-1]+(t-gcc[i-1])/s
    return float('nan')

rng=np.random.default_rng(SEED); res={}
print("Gabriel N=17000 ...", flush=True)
G=gabriel_graph(17000,rng); k=2*G.number_of_edges()/G.number_of_nodes()
cs,gcc=percolation_curve(G,15,100,rng); res['gabriel']={'k':k,'c':cs.tolist(),'gcc':gcc.tolist(),'cstar':threshold_at(cs,gcc)}
print(f"  c*={res['gabriel']['cstar']:.4f} <k>={k:.2f}", flush=True)
res['cities']={}
for name in CITIES:
    G=load_graph(name); k=2*G.number_of_edges()/G.number_of_nodes()
    cs,gcc=percolation_curve(G,10,80,rng); cst=threshold_at(cs,gcc)
    res['cities'][name]={'k':k,'N':G.number_of_nodes(),'E':G.number_of_edges(),'c':cs.tolist(),'gcc':gcc.tolist(),'cstar':cst}
    print(f"  {name:<14} N={G.number_of_nodes():>7,} <k>={k:.2f} c*={cst:.3f}", flush=True)
json.dump(res, open(os.path.join(OUT,'percolation_results.json'),'w'))

# ---- figure (PNAS double-column size) ----
FS_LABEL,FS_TICK,FS_LEG,FS_PANEL=8,7,5.5,10
fig,axes=plt.subplots(1,3,figsize=(7.0,2.45)); ax1,ax2,ax3=axes
g=res['gabriel']
ax1.plot(np.array(g['c'])*100,g['gcc'],lw=1.6,color='#1f77b4',label=f"Gabriel, $N$ = 17,000\n$\\langle k \\rangle$ = {g['k']:.2f}")
ax1.axvline(C_STAR*100,color='red',ls='--',lw=1.1,alpha=0.75,label=f"$c^*_{{\\rm bond}}$ = {C_STAR}")
ax1.set_xlabel('Coverage ratio $c$ (%)',fontsize=FS_LABEL); ax1.set_ylabel(r'Giant component  $P_\infty$',fontsize=FS_LABEL)
ax1.legend(fontsize=FS_LEG,loc='lower right',framealpha=0.9,handlelength=1.3,borderpad=0.35)
order=sorted(res['cities'],key=lambda n:res['cities'][n]['cstar']); cmap=plt.cm.viridis; handles=[]
for i,name in enumerate(order):
    r=res['cities'][name]
    (ln,)=ax2.plot(np.array(r['c'])*100,r['gcc'],lw=1.1,alpha=0.9,color=cmap(i/max(len(order)-1,1)),label=f"{NAMES.get(name,name)} ({r['cstar']:.2f})")
    handles.append(ln)
ax2.axvline(C_STAR*100,color='red',ls='--',lw=1.1,alpha=0.75)
ax2.set_xlabel('Coverage ratio $c$ (%)',fontsize=FS_LABEL); ax2.set_ylabel(r'$P_\infty$',fontsize=FS_LABEL)
ks=np.array([res['cities'][n]['k'] for n in order]); cst=np.array([res['cities'][n]['cstar'] for n in order])
for i,name in enumerate(order):
    ax3.scatter(res['cities'][name]['k'],res['cities'][name]['cstar'],s=34,color=cmap(i/max(len(order)-1,1)),edgecolors='k',linewidths=0.5,zorder=5)
a_fit=float(np.mean(cst*ks)); kk=np.linspace(2.6,4.15,100)
ax3.plot(kk,a_fit/kk,'k--',lw=1.1,alpha=0.6,label=f'$c^* \\approx {a_fit:.2f}/\\langle k \\rangle$')
ax3.scatter([g['k']],[g['cstar']],marker='*',s=130,color='red',edgecolors='k',linewidths=0.5,zorder=6,label='Gabriel model')
ax3.set_xlabel(r'Mean degree  $\langle k \rangle$',fontsize=FS_LABEL); ax3.set_ylabel(r'Measured threshold  $c^*$',fontsize=FS_LABEL)
ax3.legend(fontsize=FS_LEG,loc='upper right',framealpha=0.9,handlelength=1.3,borderpad=0.35)
for ax,lab in zip(axes,'ABC'):
    ax.grid(True,alpha=0.22,lw=0.5); ax.tick_params(labelsize=FS_TICK,length=2.5,width=0.6)
    for s in ax.spines.values(): s.set_linewidth(0.7)
    ax.text(-0.20,1.08,lab,transform=ax.transAxes,fontsize=FS_PANEL,fontweight='bold',va='top',ha='left')
ax1.set_xlim(0,102); ax1.set_ylim(0,1.05); ax2.set_xlim(0,102); ax2.set_ylim(0,1.05)
plt.tight_layout(pad=0.5,w_pad=1.6,rect=[0,0.13,1,1])
fig.legend(handles=handles,labels=[h.get_label() for h in handles],loc='lower center',ncol=5,fontsize=FS_LEG,frameon=False,handlelength=1.3,columnspacing=1.3,labelspacing=0.3,bbox_to_anchor=(0.5,-0.005))
fig.savefig(os.path.join(OUT,'fig_percolation_realcities.pdf'),dpi=300)
res['fit_a']=a_fit; json.dump(res, open(os.path.join(OUT,'percolation_results.json'),'w'))
print(f"fit c* ~ {a_fit:.3f}/<k>; range {cst.min():.3f}-{cst.max():.3f}; saved figure")
