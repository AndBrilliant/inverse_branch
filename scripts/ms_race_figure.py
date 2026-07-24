"""Generator for the ms discrimination figure (adapts the companion's Fig. 2).
Reads ms_history.csv from the Soddy-paper repository (columns:
year, value_2GeV_MeV, err_MeV, source) -- historical PDG/FLAG determinations
of ms(2 GeV). Plots the points with error bars plus TWO prediction lines:
  cone-exact (this paper):      92.65 MeV
  companion closed form:        93.45 MeV  (alpha^2 * mu_star run to 2 GeV)
and a shaded +-0.3 MeV band marking the anticipated lattice precision at which
the two rival exact claims separate (Tests, item 5).
No data points are fabricated here: supply the CSV from the archived repo."""
import sys, csv
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
CONE, TOWER = 92.65, 93.45
rows=list(csv.DictReader(open(sys.argv[1] if len(sys.argv)>1 else "ms_history.csv")))
yr=[float(r['year']) for r in rows]; v=[float(r['value_2GeV_MeV']) for r in rows]
e=[float(r['err_MeV']) for r in rows]
fig,ax=plt.subplots(figsize=(5.2,3.4))
ax.errorbar(yr,v,yerr=e,fmt='s',ms=4,color='k',capsize=2,lw=1)
ax.axhline(CONE,color='0.2',ls='-',lw=1.2); ax.annotate('cone-exact 92.65',(yr[0],CONE),
    textcoords='offset points',xytext=(0,-11),fontsize=8)
ax.axhline(TOWER,color='0.2',ls='--',lw=1.2); ax.annotate('companion 93.45',(yr[0],TOWER),
    textcoords='offset points',xytext=(0,5),fontsize=8)
ax.axhspan(CONE-0.3,CONE+0.3,color='0.9',zorder=0)
ax.set_xlabel('year'); ax.set_ylabel(r'$m_s(2\,\mathrm{GeV})$ [MeV]')
plt.tight_layout(); plt.savefig('fig_ms_race.pdf',bbox_inches='tight')
print("wrote fig_ms_race.pdf")
