"""One fixed channel, several chemical potentials, on the same axes.

The axes stay what they were: external momentum along Gamma-X-M-Gamma
across, Pi(Q) or chi(q) up. Each curve is one value of mu, so the figure
shows how filling moves a single channel rather than how the channels
compare at one filling.

Everything is reused from lieb_simple.py; only mu is varied, by setting
the constant there before each sweep. Run: python3 lieb_vs_mu.py
"""
import numpy as np
import matplotlib.pyplot as plt

import lieb_simple as L

# ---------------------------------------------------------------------
# What to plot. A channel is a bubble, a spin pair and a symmetry.
# ---------------------------------------------------------------------
#   kind "PP" -> Pi(Q)               kind "PH" -> chi(q)
#   spin "opposite" -> singlet, T0   spin "equal" -> T+-1 (PP)
#                   -> chi_up,down                 -> chi_up,up  (PH)
#   form "s", "px" or "py"
CHANNELS = [
    ("PP", "opposite", "s"),    # Singlet s-wave pairing.
    ("PH", "equal", "s"),       # Equal-spin charge-like response.
]

# Chemical potentials, in units of the hopping. The upper band runs from
# 4*lambda = 0.8 t at M up to 2*sqrt(2) t = 2.83 t at Gamma, so these
# five sweep the band from just above its bottom to past its middle.
MU_OVER_T = [0.9, 1.2, 1.5, 1.8, 2.1]

L.NK = 120       # Same grid knobs as lieb_simple.
L.NSEG = 60
L.UPPER_BAND_ONLY = True

# One hue, light to dark, because mu is ordered.
RAMP = ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def sweep_at_mu(mu, kind, spin, form):
    """One curve: the channel along the path, at this chemical potential.

    lieb_simple reads mu from its module constant, so set it and rebuild
    nothing else -- the grid and the weights do not depend on mu.
    """
    L.MU = mu
    spin2 = L.UP if spin == "equal" else L.DOWN
    points, distance, corners = L.path()
    values = np.array([L.bubble(qx, qy, kind, L.UP, spin2)[form]
                       for qx, qy in points])
    return distance, corners, values


def figure_for_channel(kind, spin, form):
    """All the chemical potentials for one channel, on one pair of axes."""
    symbol = r"\Pi(\mathbf{Q})" if kind == "PP" else r"\chi(\mathbf{q})"
    momentum = r"\mathbf{Q}" if kind == "PP" else r"\mathbf{q}"
    print(f"{kind} {spin} {form}:", flush=True)

    figure, axes = plt.subplots(figsize=(9.5, 5.4), layout="constrained")
    for color, ratio in zip(RAMP, MU_OVER_T):
        mu = ratio * L.T
        distance, corners, values = sweep_at_mu(mu, kind, spin, form)
        axes.plot(distance, values, color=color, lw=2.0,
                  label=rf"$\mu = {ratio:g}\,t$ ({mu:.2f} eV)")
        print(f"  mu = {ratio:g} t: peak {values.max():.4g} "
              f"at {'Gamma' if values.argmax() in (0, len(values)-1) else 'the path interior'}",
              flush=True)

    for corner in corners[1:-1]:
        axes.axvline(corner, color="#d8d8d5", lw=0.8, zorder=0)
    axes.set_xticks(corners, [r"$\Gamma$", "$X$", "$M$", r"$\Gamma$"])
    axes.set_xlim(corners[0], corners[-1])
    axes.set_ylim(bottom=0)
    axes.set_xlabel(rf"External momentum ${momentum}$")
    axes.set_ylabel(rf"${symbol}$  [$\mathrm{{eV}}^{{-1}}\,\mathrm{{\AA}}^{{-2}}$]")
    axes.set_title(f"{'Pairing' if kind == 'PP' else 'Particle-hole'} channel: "
                   f"{spin} spin, {form}-wave", loc="left", weight="bold", pad=10)
    axes.grid(axis="y", color="#ececea", lw=0.7)
    for edge in ("top", "right"):
        axes.spines[edge].set_visible(False)
    axes.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False,
                title="Chemical potential", borderaxespad=0)

    name = f"{'pi' if kind == 'PP' else 'chi'}_{spin}_{form}_vs_mu.png"
    figure.savefig(name)
    return name, figure


def main():
    print(f"Lieb lattice: t={L.T} eV, a0={L.A0} A, lambda={L.LAM:.3g} eV, "
          f"kT={L.KT:.3g} eV, {L.NK}x{L.NK} grid")
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 13,
                         "savefig.dpi": 180})
    written = [figure_for_channel(*channel)[0] for channel in CHANNELS]
    print("Wrote " + ", ".join(written))
    return written


if __name__ == "__main__":
    main()
