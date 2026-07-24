"""Generator for Fig. 1 (fig_branches.pdf/.png) of the inverse-branch paper.
Panels: (a) self-dual waveform Z/s = 1+sqrt(2)cos(phi+2/9) with the three family
samples; (b) direct-branch lepton ladder, m ~ |Z_a|^2; (c) seesaw inversion:
partner ladder M_a ~ |Z_a|^2 with arrows to the inverted light ladder
m_a = m_D^2/M_a (heaviest partner -> lightest quark).
Deterministic (no RNG). Reproduces the committed figure exactly."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = 2/9
phi = np.linspace(0, 2*np.pi, 600)
Z = 1 + np.sqrt(2)*np.cos(phi + d)
za = np.array([1 + np.sqrt(2)*np.cos(d + 2*np.pi*a/3) for a in range(3)])
pa = np.array([2*np.pi*a/3 for a in range(3)])

fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.1),
                        gridspec_kw={'width_ratios': [1.15, 1, 1]})
ax = axs[0]
ax.plot(phi, Z, 'k-', lw=1.4)
ax.axhline(0, color='0.7', lw=0.6)
ax.plot(pa, za, 'o', ms=7, mfc='w', mec='k', mew=1.4, zorder=5)
for p, z, lab in zip(pa, za, [r'$a{=}0$', r'$a{=}1$', r'$a{=}2$']):
    ax.annotate(lab, (p, z), textcoords="offset points", xytext=(6, 7), fontsize=9)
ax.set_xlabel(r'$\varphi$'); ax.set_ylabel(r'$Z(\varphi)/s$')
ax.set_xticks([0, 2*np.pi/3, 4*np.pi/3, 2*np.pi])
ax.set_xticklabels(['0', r'$2\pi/3$', r'$4\pi/3$', r'$2\pi$'])
ax.set_title('(a) self-dual amplitude, three family samples', fontsize=9.5)

ax = axs[1]
mlep = za**2/np.sum(za**2)
for z2, lab in zip(sorted(mlep), ['$e$', '$\\mu$', '$\\tau$']):
    ax.hlines(z2, 0.25, 0.75, color='k', lw=2)
    ax.annotate(lab, (0.78, z2), fontsize=10, va='center')
ax.set_yscale('log'); ax.set_xlim(0, 1.05); ax.set_xticks([])
ax.set_ylabel(r'$m \propto |Z_a|^2$   (direct branch)')
ax.set_title('(b) leptons: direct coupling', fontsize=9.5)

ax = axs[2]
Mpart = za**2/np.sum(za**2)
mlight = (1/za**2); mlight /= mlight.max()*8
for M, lab in zip(sorted(Mpart), ['$D_b$', '$D_s$', '$D_d$']):
    ax.hlines(M, 0.05, 0.40, color='k', lw=2)
    ax.annotate(lab, (0.005, M), fontsize=9, va='center', ha='left')
for m, lab in zip(sorted(mlight), ['$d$', '$s$', '$b$']):
    ax.hlines(m, 0.60, 0.95, color='k', lw=2)
    ax.annotate(lab, (0.97, m), fontsize=10, va='center')
for M, m in zip(sorted(Mpart, reverse=True), sorted(mlight)):
    ax.annotate('', xy=(0.58, m), xytext=(0.42, M),
                arrowprops=dict(arrowstyle='->', color='0.45', lw=1.0))
ax.set_yscale('log'); ax.set_xlim(0, 1.15); ax.set_xticks([])
ax.set_ylabel(r'$M_a \propto |Z_a|^2 \;\rightarrow\; m_a = m_D^2/M_a$')
ax.set_title(r'(c) down quarks: seesaw inversion', fontsize=9.5)

plt.tight_layout()
plt.savefig('fig_branches.pdf', bbox_inches='tight')
plt.savefig('fig_branches.png', dpi=180, bbox_inches='tight')
print("wrote fig_branches.pdf/.png")
