"""Prior-sensitivity variants of the coincidence-budget MC (caveat ii).
Same J statistic and J_obs = 1.0536e-3 as the engines; seed 314159; N = 1e7 each.
  log-uniform (paper's quoted measure, pooled engines): f = 6.3e-6
  broad log-normal (centered mid-range, sigma = 1.5 decades): 58/1e7 -> 5.8e-6
  uniform-in-mass (same ranges): 0/1e7 -> f < 3.0e-7 (CP95 upper limit)
The quoted log-uniform value is the most null-favorable of the three."""
import numpy as np
ang=2*np.pi*np.arange(3)/3; c=np.cos(ang); s=np.sin(ang)
def kdist(m):
    out=None
    for v in (np.sqrt(m),1/np.sqrt(m)):
        A=v.mean(axis=-1); X=(2/3)*(v*c).sum(-1); Y=-(2/3)*(v*s).sum(-1)
        d=np.abs(np.hypot(X,Y)/(np.sqrt(2)*A)-1)
        out=d if out is None else np.minimum(out,d)
    return out
Jobs=1.0536e-3
def run(draw,N=10_000_000,B=1_000_000,seed=314159):
    rng=np.random.default_rng(seed); h=0
    for _ in range(N//B):
        lep,dwn=draw(rng,B); h+=int(((kdist(lep)+kdist(dwn))<=Jobs).sum())
    return h,N
uni=lambda r,B:(r.uniform(0.3,2000.,(B,3)),r.uniform(2.0,10000.,(B,3)))
def logn(r,B):
    return (np.exp(r.normal(np.log(np.sqrt(0.3*2000.)),1.5*np.log(10),(B,3))),
            np.exp(r.normal(np.log(np.sqrt(2.0*10000.)),1.5*np.log(10),(B,3))))
for nm,d in [("uniform-in-mass",uni),("log-normal",logn)]:
    h,N=run(d); print(nm,h,"/",N)
