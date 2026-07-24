"""Engine B of the coincidence-budget MC (Sec. 'The coincidence budget').
Observed inputs computed in-script at full precision (PDG: me=0.51099895,
mmu=105.6583755, mtau=1776.93; downs 4.70/93.4/4966.0 at 2 GeV).
J_obs = 3.305e-6 + 1.0503e-3. Seed 20260726, N=1e7 -> 69 hits (f=6.9e-6).
Pooled with engine A (Kimi implementation, seed 20260723, N=2e7 -> 120 hits):
189/3e7, f = 6.3e-6, CP95 [5.4, 7.3]e-6. Range-widened variant: 30/4e6 = 7.5e-6."""
import numpy as np
ang=2*np.pi*np.arange(3)/3; c=np.cos(ang); s=np.sin(ang)
def kdist(m):
    out=None
    for v in (np.sqrt(m),1/np.sqrt(m)):
        A=v.mean(axis=-1); X=(2/3)*(v*c).sum(-1); Y=-(2/3)*(v*s).sum(-1)
        d=np.abs(np.hypot(X,Y)/(np.sqrt(2)*A)-1)
        out=d if out is None else np.minimum(out,d)
    return out
lep_obs=np.array([[0.51099895,105.6583755,1776.93]])
dwn_obs=np.array([[4.70,93.4,4966.0]])
Jobs=float((kdist(lep_obs)+kdist(dwn_obs))[0])
print(f"J_obs = {Jobs:.4e}")
rng=np.random.default_rng(20260726)
N,B,hits=10_000_000,1_000_000,0
for _ in range(N//B):
    lep=np.exp(rng.uniform(np.log(0.3),np.log(2000.),(B,3)))
    dwn=np.exp(rng.uniform(np.log(2.0),np.log(10000.),(B,3)))
    hits+=int(((kdist(lep)+kdist(dwn))<=Jobs).sum())
print(f"{hits}/{N} -> f = {hits/N:.2e}")
