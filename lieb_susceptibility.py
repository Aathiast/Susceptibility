"""Static pair (Pi) and particle-hole (chi) susceptibilities of the Lieb lattice.

Physical units: energies in eV, lengths in Angstrom, so that the plotted
susceptibilities are chi and Pi themselves -- no t*a0^2 rescaling -- in
1/(eV Angstrom^2) for --units per_area, or 1/eV per cell for --units per_cell.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np


# =====================================================================
# 0. Physical scales of the Lieb lattice
# =====================================================================
# Values for the CuO2-plane realization of the Lieb lattice: the Cu-O
# nearest-neighbour hopping and the Cu-Cu (square) lattice constant.
# Edit these two numbers to move to another realization; every other
# energy below is quoted as a dimensionless ratio of the hopping.
HOPPING_EV = 1.3        # t, in eV.
LATTICE_ANGSTROM = 3.8  # a0, in Angstrom.
ENERGY_UNIT = "eV"
LENGTH_UNIT = "Angstrom"

LAMBDA_OVER_T = 0.20      # Intrinsic SOC.
MU_OVER_T = 1.20          # Chemical potential measured from the flat band.
TEMPERATURE_OVER_T = 0.04  # k_B T.


# =====================================================================
# 1. Editable physical and numerical parameters
# =====================================================================
@dataclass(frozen=True)
class Parameters:
    t: float = HOPPING_EV                              # eV.
    lam: float = LAMBDA_OVER_T * HOPPING_EV            # eV.
    mu: float = MU_OVER_T * HOPPING_EV                 # eV.
    temperature: float = TEMPERATURE_OVER_T * HOPPING_EV  # eV; must be positive.
    a0: float = LATTICE_ANGSTROM                       # Angstrom.
    nk: int = 160              # nk x nk midpoint grid over the full BZ.
    nseg: int = 60             # External momentum intervals per path segment.
    bands: str = "upper"      # "upper" or "all" (nine ordered band pairs).
    forms: str = "lattice"    # "lattice": sin(a0*k); "continuum": k.
    normalize_forms: bool = True
    pp_normalization: str = "sim"
    units: str = "per_area"

    def validate(self):
        if not np.all(np.isfinite([self.t, self.lam, self.mu,
                                   self.temperature, self.a0])):
            raise ValueError("All physical parameters must be finite.")
        if self.t <= 0 or self.a0 <= 0 or self.temperature <= 0:
            raise ValueError("Require t > 0, a0 > 0, and temperature > 0.")
        if self.nk < 8 or self.nseg < 1:
            raise ValueError("Require nk >= 8 and nseg >= 1.")
        for value, allowed in [
            (self.bands, ("upper", "all")),
            (self.forms, ("lattice", "continuum")),
            (self.pp_normalization, ("sim", "appendix")),
            (self.units, ("per_area", "per_cell")),
        ]:
            if value not in allowed:
                raise ValueError(f"Expected one of {allowed}; received {value!r}.")


DEFAULTS = Parameters()
SPINS = (+1, -1)  # Array indices 0,1 correspond to up,down.
FORM_NAMES = ("s", "px", "py", "pplus", "pminus")


# =====================================================================
# 2. Hamiltonian coefficients and spectral projectors
# =====================================================================
def coefficients(momentum, p):
    """Accept an (N,2) momentum array. Do not fold shifted momenta."""
    momentum = np.asarray(momentum, dtype=float)
    kx, ky = momentum[:, 0], momentum[:, 1]
    ax = -2.0 * p.t * np.cos(0.5 * p.a0 * kx)
    ay = -2.0 * p.t * np.cos(0.5 * p.a0 * ky)
    az = (4.0 * p.lam * np.sin(0.5 * p.a0 * kx)
          * np.sin(0.5 * p.a0 * ky))
    a = np.sqrt(ax**2 + ay**2 + az**2)
    return ax, ay, az, a


def normal_hamiltonian(momentum, p):
    """h_s(k) in the (B,A,C) basis; shape (2,N,3,3), excluding -mu I."""
    ax, ay, az, a = coefficients(momentum, p)
    h = np.zeros((2, len(a), 3, 3), dtype=complex)
    h[:, :, 0, 1] = h[:, :, 1, 0] = ax
    h[:, :, 1, 2] = h[:, :, 2, 1] = ay
    h[:, :, 0, 2] = -1j * np.asarray(SPINS)[:, None] * az[None, :]
    h[:, :, 2, 0] = +1j * np.asarray(SPINS)[:, None] * az[None, :]
    return h, a


def band_projectors(momentum, p):
    """Return dictionaries P[band] and xi[band], for band=+1 or -1,0,+1.

    P has shape (spin, momentum, orbital, orbital).
    xi has shape (momentum,), because the spectrum is spin degenerate.
    T_s = h_s/a, P_+ = (T_s^2+T_s)/2, P_0 = I-T_s^2,
    and P_- = (T_s^2-T_s)/2.
    """
    h, a = normal_hamiltonian(momentum, p)
    if np.any(a <= 64 * np.finfo(float).eps * max(p.t, abs(p.lam))):
        raise ValueError(
            "A sampled momentum has a=0, where individual band projectors "
            "are undefined. Use a gapped SOC model or change the mesh."
        )
    unit_h = h / a[None, :, None, None]
    unit_h_squared = unit_h @ unit_h
    projectors = {+1: 0.5 * (unit_h_squared + unit_h)}
    energies = {+1: a - p.mu}
    if p.bands == "all":
        projectors[0] = np.eye(3)[None, None, :, :] - unit_h_squared
        projectors[-1] = 0.5 * (unit_h_squared - unit_h)
        energies[0] = np.full_like(a, -p.mu)
        energies[-1] = -a - p.mu
    return projectors, energies


def shifted_projectors(k, external_momentum, p, bubble):
    """Build the two projectors explicitly as functions of k and Q or q."""
    external_momentum = np.asarray(external_momentum, dtype=float)
    k1 = k + external_momentum / 2.0
    if bubble == "PP":
        k2 = -k + external_momentum / 2.0
    elif bubble == "PH":
        k2 = k - external_momentum / 2.0
    else:
        raise ValueError("bubble must be 'PP' or 'PH'.")
    # The real cos(k/2),sin(k/2) Bloch gauge is not elementwise periodic.
    # Evaluate these momenta directly; independent modulo folding would
    # require the associated orbital gauge transformations on the vertices.
    P1, xi1 = band_projectors(k1, p)
    P2, xi2 = band_projectors(k2, p)
    return P1, xi1, P2, xi2


# =====================================================================
# 3. Matrix products and orbital traces -- the requested W calculation
# =====================================================================
def coherence_trace(P1, P2, bubble):
    """W[s,s',k] for identity orbital vertices, retaining both spin labels."""
    if bubble == "PP":
        # Ordinary transpose on the orbital indices; NOT Hermitian conjugate.
        second_matrix = np.swapaxes(P2, -1, -2)
    elif bubble == "PH":
        second_matrix = P2
    else:
        raise ValueError("bubble must be 'PP' or 'PH'.")
    # Shapes: (2,1,N,3,3) @ (1,2,N,3,3) -> (2,2,N,3,3).
    matrix_product = P1[:, None, ...] @ second_matrix[None, :, ...]
    W = np.trace(matrix_product, axis1=-2, axis2=-1)
    if np.max(np.abs(W.imag)) > 1e-10:
        raise ArithmeticError("Identity-vertex projector trace should be real.")
    return W.real


# =====================================================================
# 4. Stable, static Matsubara kernels
# =====================================================================
def minus_fermi_derivative(energy, temperature):
    """-n_F'(energy), evaluated without overflowing cosh."""
    z = np.exp(-np.abs(np.asarray(energy)) / temperature)
    return z / (temperature * (1.0 + z)**2)


def kernel_ph(x, y, temperature):
    """[n_F(y)-n_F(x)]/(x-y); coincident limit is -n_F'((x+y)/2)."""
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    delta = x - y
    numerator = 0.5 * (np.tanh(x / (2 * temperature))
                       - np.tanh(y / (2 * temperature)))
    near = np.abs(delta) <= 1e-6 * temperature
    answer = np.empty_like(delta)
    np.divide(numerator, delta, out=answer, where=~near)
    answer[near] = minus_fermi_derivative(
        0.5 * (x[near] + y[near]), temperature
    )
    return answer


def kernel_pp(x, y, temperature):
    """[1-n_F(x)-n_F(y)]/(x+y); handles x+y=0 analytically.

    Exactly K_PP(x,y)=K_PH(x,-y).
    In particular K_PP(x,x)=tanh[x/(2T)]/(2x), with value 1/(4T) at x=0.
    """
    return kernel_ph(x, -np.asarray(y), temperature)


# =====================================================================
# 5. BZ grid, momentum form factors, and normalized spin-channel matrices
# =====================================================================
def brillouin_zone_grid(p):
    """Midpoints in [-pi/a0,pi/a0)^2, with no duplicated BZ boundary."""
    one_axis = (2 * np.pi / p.a0) * (
        (np.arange(p.nk) + 0.5) / p.nk - 0.5
    )
    kx, ky = np.meshgrid(one_axis, one_axis, indexing="ij")
    return np.column_stack((kx.ravel(), ky.ravel()))


def momentum_form_factors(k, p):
    """Columns s,px,py,p+,p-; optionally normalize mean_BZ |f|^2 to one.

    Default normalized lattice forms:
      f_s=1, f_px=sqrt(2)sin(a0*kx), f_py=sqrt(2)sin(a0*ky),
      f_p+ = sin(a0*kx)+i sin(a0*ky), f_p- = f_p+^*.

    --forms continuum --raw-forms reproduces 1,kx,ky,kx+i*ky,kx-i*ky
    from the Sim note, using this fixed BZ representative for k.
    Those polynomial forms are not periodic lattice harmonics, and with
    a0 in Angstrom they also carry inverse-length units.
    """
    if p.forms == "lattice":
        px, py = np.sin(p.a0 * k[:, 0]), np.sin(p.a0 * k[:, 1])
    else:
        px, py = k[:, 0], k[:, 1]
    F = np.column_stack((np.ones(len(k)), px, py,
                         px + 1j * py, px - 1j * py))
    if p.normalize_forms:
        F /= np.sqrt(np.mean(np.abs(F)**2, axis=0))[None, :]
    return F


@dataclass
class Channel:
    name: str
    form: str
    spin_matrix: np.ndarray


def define_channels():
    """PP: S_s and T0,T+1,T-1 with px,py,p+,p-.

    PH equal: I_2/sqrt(2), averaging the two spin-preserving transitions.
    PH opposite: s_x/sqrt(2), averaging the two spin-flip transitions.
    s_z/sqrt(2) and s_y/sqrt(2) have the same diagonal responses, respectively,
    in this spin-degenerate identity-orbital model.
    """
    rt2 = np.sqrt(2.0)
    pair_spin = {
        "S": np.array([[0, 1], [-1, 0]], complex) / rt2,
        "T0": np.array([[0, 1], [1, 0]], complex) / rt2,
        "Tup": np.array([[1, 0], [0, 0]], complex),
        "Tdown": np.array([[0, 0], [0, 1]], complex),
    }
    particle_hole_spin = {
        "equal": np.eye(2, dtype=complex) / rt2,
        "opposite": np.array([[0, 1], [1, 0]], complex) / rt2,
    }
    pp = [Channel("S_s", "s", pair_spin["S"])]
    pp += [Channel(f"{spin}_{form}", form, pair_spin[spin])
           for spin in ("T0", "Tup", "Tdown") for form in FORM_NAMES[1:]]
    ph = [Channel(f"{spin}_{form}", form, particle_hole_spin[spin])
          for spin in ("equal", "opposite") for form in FORM_NAMES]
    return pp, ph


def susceptibility_unit(p, latex=False):
    """Units of the plotted chi and Pi, given eV energies and Angstrom lengths."""
    if p.units == "per_area":
        return (rf"\mathrm{{{ENERGY_UNIT}}}^{{-1}}\,\mathrm{{\AA}}^{{-2}}"
                if latex else f"1/({ENERGY_UNIT} {LENGTH_UNIT}^2)")
    return (rf"\mathrm{{{ENERGY_UNIT}}}^{{-1}}\,\mathrm{{cell}}^{{-1}}"
            if latex else f"1/{ENERGY_UNIT} per cell")


# =====================================================================
# 6. Full BZ integration of the susceptibility matrix mu,nu
# =====================================================================
class SusceptibilitySolver:
    def __init__(self, parameters=DEFAULTS):
        parameters.validate()
        self.p = parameters
        self.k = brillouin_zone_grid(parameters)
        self.F = momentum_form_factors(self.k, parameters)
        self.pp_channels, self.ph_channels = define_channels()
        # f_l^*(k) f_l'(k), retaining complex phases for off-diagonal channels.
        self.form_products = (
            self.F.conj()[:, :, None] * self.F[:, None, :]
        ).reshape(len(self.k), len(FORM_NAMES)**2)
        # integral_BZ d^2k/(2pi)^2 = mean_BZ/a0^2.
        self.integration_factor = (
            1.0 / parameters.a0**2 if parameters.units == "per_area" else 1.0
        )

    def bubble_matrix(self, external_momentum, bubble):
        p = self.p
        P1, xi1, P2, xi2 = shifted_projectors(
            self.k, external_momentum, p, bubble
        )
        kernel = kernel_pp if bubble == "PP" else kernel_ph
        weighted_trace = np.zeros((2, 2, len(self.k)), dtype=float)
        # Default: only (+,+). With --bands all: all nine ordered band pairs.
        for u in P1:
            for r in P2:
                W = coherence_trace(P1[u], P2[r], bubble)
                K = kernel(xi1[u], xi2[r], p.temperature)
                weighted_trace += W * K[None, None, :]

        # Integrate each spin pair and each f_l^* f_l' over the full BZ.
        form_integrals = (
            weighted_trace.reshape(4, -1) @ self.form_products
        ).reshape(2, 2, len(FORM_NAMES), len(FORM_NAMES))
        form_integrals *= self.integration_factor / len(self.k)

        channels = self.pp_channels if bubble == "PP" else self.ph_channels
        spin = np.stack([ch.spin_matrix for ch in channels])
        index = np.asarray([FORM_NAMES.index(ch.form) for ch in channels])
        result = np.zeros((len(channels), len(channels)), dtype=complex)
        for s in range(2):
            for sp in range(2):
                spin_weight = spin[:, s, sp].conj()[:, None] * spin[:, s, sp][None, :]
                orbital_integral = form_integrals[s, sp][
                    index[:, None], index[None, :]
                ]
                result += spin_weight * orbital_integral
        if bubble == "PP" and p.pp_normalization == "appendix":
            result *= 0.5
        return result

    def solve(self, path, progress=True):
        pair = np.empty((len(path), len(self.pp_channels),
                         len(self.pp_channels)), complex)
        ph = np.empty((len(path), len(self.ph_channels),
                       len(self.ph_channels)), complex)
        start = perf_counter()
        for i, external in enumerate(path):
            pair[i] = self.bubble_matrix(external, "PP")
            ph[i] = self.bubble_matrix(external, "PH")
            if progress and (i % max(1, len(path)//10) == 0 or i == len(path)-1):
                print(f"{i+1:4d}/{len(path)} momenta; "
                      f"{perf_counter()-start:.1f} s", flush=True)
        return pair, ph


def gamma_x_m_gamma(p):
    """External path; both Q and q use these same coordinates."""
    nodes = (np.pi / p.a0) * np.array(
        [[0, 0], [1, 0], [1, 1], [0, 0]], dtype=float
    )
    path = np.concatenate([
        np.linspace(nodes[i], nodes[i+1], p.nseg, endpoint=False)
        for i in range(3)
    ] + [nodes[-1:]], axis=0)
    distance = np.r_[0.0, np.cumsum(
        np.linalg.norm(np.diff(path, axis=0), axis=1) * p.a0
    )]
    tick_indices = np.arange(4) * p.nseg
    return path, distance, distance[tick_indices]


# =====================================================================
# 7. Two figures; degenerate curves share a legend entry
# =====================================================================
def make_plots(p, distance, ticks, pair, ph, pp_channels, ph_channels, out):
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 13, "axes.titlesize": 15,
        "legend.fontsize": 10, "xtick.labelsize": 13, "ytick.labelsize": 11,
        "savefig.dpi": 180,
    })
    colors = {"s": "#252936", "px": "#2166ac",
              "py": "#ce5a26", "pplus": "#23856d"}
    forms = {"s": "s", "px": "p_x", "py": "p_y", "pplus": r"p_\pm"}
    # chi and Pi are plotted as they are, in the physical units set by
    # t in eV and a0 in Angstrom. No t*a0^2 rescaling.
    unit_label = susceptibility_unit(p, latex=True)
    band_text = "upper band" if p.bands == "upper" else "all three bands"
    parameter_text = (
        rf"Lieb lattice, {band_text}; $t={p.t:g}$ eV, "
        rf"$a_0={p.a0:g}$ $\mathrm{{\AA}}$; "
        rf"$\lambda={p.lam:g}$ eV, $\mu={p.mu:g}$ eV, "
        rf"$k_BT={p.temperature:g}$ eV; BZ grid $ {p.nk}\times{p.nk} $"
    )
    figures = []
    for kind, matrix, channels in [
        ("PP", pair, pp_channels), ("PH", ph, ph_channels)
    ]:
        fig, ax = plt.subplots(figsize=(11.8, 6.1), layout="constrained")
        index = {ch.name: i for i, ch in enumerate(channels)}
        if kind == "PP":
            ax.plot(distance, matrix[:, index["S_s"], index["S_s"]].real,
                    color=colors["s"], lw=2.4, label=r"$S,\ s$")
            for spin, spin_label, style in [
                ("T0", "T_0", "-"), ("Tup", r"T_{+1}=T_{-1}", "--")
            ]:
                for form in ("px", "py", "pplus"):
                    i = index[f"{spin}_{form}"]
                    ax.plot(distance, matrix[:, i, i].real,
                            color=colors[form], ls=style, lw=2.0,
                            label=rf"$ {spin_label},\ {forms[form]} $")
            title = "Particle-particle susceptibility"
            ylabel = (rf"$ \Pi_{{\mu\mu}}(\mathbf{{Q}}) $"
                      rf"  $ [{unit_label}] $")
            xlabel = r"Total pair momentum $\mathbf{Q}$ along $\Gamma-X-M-\Gamma$"
            filename = "pair_susceptibility.png"
            legend_title = f"Channels ({p.pp_normalization} normalization)"
        else:
            for spin, spin_label, style in [
                ("equal", r"\mathrm{equal}", "-"),
                ("opposite", r"\mathrm{opposite}", "--"),
            ]:
                for form in ("s", "px", "py", "pplus"):
                    i = index[f"{spin}_{form}"]
                    ax.plot(distance, matrix[:, i, i].real,
                            color=colors[form], ls=style, lw=2.0,
                            label=rf"$ {spin_label},\ {forms[form]} $")
            title = "Particle-hole susceptibility"
            ylabel = (rf"$ \chi_{{\mu\mu}}(\mathbf{{q}}) $"
                      rf"  $ [{unit_label}] $")
            xlabel = r"Transfer momentum $\mathbf{q}$ along $\Gamma-X-M-\Gamma$"
            filename = "particle_hole_susceptibility.png"
            legend_title = "Spin transitions"
        for tick in ticks[1:-1]:
            ax.axvline(tick, color="#c5cad0", lw=0.9, zorder=0)
        ax.set_xticks(ticks, [r"$\Gamma$", "$X$", "$M$", r"$\Gamma$"])
        ax.set_xlim(distance[0], distance[-1])
        ax.set_ylim(bottom=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=34, weight="bold")
        ax.text(0, 1.025, parameter_text, transform=ax.transAxes,
                fontsize=9.5, color="#555b63")
        ax.grid(axis="y", color="#e1e5e9", lw=0.65)
        for edge in ("top", "right"):
            ax.spines[edge].set_visible(False)
        ax.legend(loc="upper left", bbox_to_anchor=(1.015, 1),
                  frameon=False, title=legend_title, borderaxespad=0)
        fig.savefig(out / filename)
        figures.append(fig)
    return figures


def save_data(out, p, path, distance, ticks, pair, ph, pp_channels, ph_channels):
    """NPZ stores full complex matrices; CSVs store every diagonal channel."""
    pp_names = [ch.name for ch in pp_channels]
    ph_names = [ch.name for ch in ph_channels]
    unit = susceptibility_unit(p)
    np.savez_compressed(
        out / "susceptibility_matrices.npz",
        external_momentum=path, path_distance=distance, path_ticks=ticks,
        Pi=pair, Chi=ph, pp_channels=np.asarray(pp_names),
        ph_channels=np.asarray(ph_names),
        susceptibility_unit=np.asarray(unit),
        energy_unit=np.asarray(ENERGY_UNIT),
        length_unit=np.asarray(LENGTH_UNIT),
        parameters_json=np.asarray(json.dumps(asdict(p))),
    )
    for filename, matrix, names in [
        ("pair_susceptibility.csv", pair, pp_names),
        ("particle_hole_susceptibility.csv", ph, ph_names),
    ]:
        with (out / filename).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([f"# susceptibility columns in {unit}; "
                             f"momenta in 1/{LENGTH_UNIT}"])
            writer.writerow(["path_distance_a0", "momentum_x", "momentum_y"] + names)
            diagonal = np.diagonal(matrix, axis1=-2, axis2=-1).real
            writer.writerows(np.column_stack((distance, path, diagonal)))
    (out / "parameters.json").write_text(json.dumps(
        {**asdict(p), "energy_unit": ENERGY_UNIT,
         "length_unit": LENGTH_UNIT, "susceptibility_unit": unit},
        indent=2) + "\n")


# =====================================================================
# 8. Entry points: importable (notebook) and command line
# =====================================================================
def build_parameters(t=None, a0=None, lam=None, mu=None,
                     temperature=None, **rest):
    """Parameters with the energies tracking t whenever they are omitted."""
    t = DEFAULTS.t if t is None else float(t)
    return replace(
        DEFAULTS,
        t=t,
        a0=DEFAULTS.a0 if a0 is None else float(a0),
        lam=LAMBDA_OVER_T * t if lam is None else float(lam),
        mu=MU_OVER_T * t if mu is None else float(mu),
        temperature=(TEMPERATURE_OVER_T * t if temperature is None
                     else float(temperature)),
        **rest,
    )


def running_in_notebook():
    """True inside a Jupyter/IPython kernel, whose sys.argv holds -f kernel.json."""
    try:
        shell = sys.modules["IPython"].get_ipython()
    except (KeyError, AttributeError):
        return False
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


def run(out="lieb_results", save=True, plot=True, progress=True,
        parameters=None, **overrides):
    """Notebook entry point, e.g. run(nk=80, mu=1.0, bands="all").

    Accepts any Parameters field as a keyword; t, a0, lam, mu and
    temperature are in eV and Angstrom. Returns a dictionary with the
    parameters, the path, the full complex matrices and the figures.
    Figures are left open so that Jupyter displays them inline.
    """
    if parameters is not None and overrides:
        raise TypeError("Pass either parameters=... or keyword overrides.")
    p = parameters if parameters is not None else build_parameters(**overrides)
    p.validate()
    solver = SusceptibilitySolver(p)
    path, distance, ticks = gamma_x_m_gamma(p)
    pair, ph = solver.solve(path, progress=progress)
    figures = []
    out = Path(out)
    if save or plot:
        out.mkdir(parents=True, exist_ok=True)
    if save:
        save_data(out, p, path, distance, ticks, pair, ph,
                  solver.pp_channels, solver.ph_channels)
    if plot:
        figures = make_plots(p, distance, ticks, pair, ph,
                             solver.pp_channels, solver.ph_channels, out)
    return {
        "parameters": p, "unit": susceptibility_unit(p),
        "external_momentum": path, "path_distance": distance,
        "path_ticks": ticks, "Pi": pair, "Chi": ph,
        "pp_channels": solver.pp_channels, "ph_channels": solver.ph_channels,
        "figures": figures, "out": out,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--t", type=float, default=None,
                        help=f"Hopping in {ENERGY_UNIT} (default {HOPPING_EV:g}).")
    parser.add_argument("--a0", type=float, default=None,
                        help=f"Lattice constant in {LENGTH_UNIT} "
                             f"(default {LATTICE_ANGSTROM:g}).")
    parser.add_argument("--lam", type=float, default=None,
                        help=f"Intrinsic SOC in {ENERGY_UNIT} "
                             f"(default {LAMBDA_OVER_T:g}*t).")
    parser.add_argument("--mu", type=float, default=None,
                        help=f"Chemical potential in {ENERGY_UNIT} "
                             f"(default {MU_OVER_T:g}*t).")
    parser.add_argument("--temperature", type=float, default=None,
                        help=f"k_B T in {ENERGY_UNIT} "
                             f"(default {TEMPERATURE_OVER_T:g}*t).")
    parser.add_argument("--nk", type=int, default=DEFAULTS.nk)
    parser.add_argument("--nseg", type=int, default=DEFAULTS.nseg)
    parser.add_argument("--bands", choices=["upper", "all"], default=DEFAULTS.bands)
    parser.add_argument("--forms", choices=["lattice", "continuum"], default=DEFAULTS.forms)
    parser.add_argument("--raw-forms", action="store_true",
                        help="Disable BZ form-factor normalization.")
    parser.add_argument("--pp-normalization", choices=["sim", "appendix"],
                        default=DEFAULTS.pp_normalization)
    parser.add_argument("--units", choices=["per_area", "per_cell"], default=DEFAULTS.units)
    parser.add_argument("--out", type=Path, default=Path("lieb_results"))
    parser.add_argument("--show", action="store_true", help="Open both figures after saving.")
    if argv is None:
        # A Jupyter kernel puts its own -f kernel-....json in sys.argv,
        # which argparse would reject; run the defaults there instead.
        argv = [] if running_in_notebook() else sys.argv[1:]
    args = parser.parse_args(argv)
    p = build_parameters(
        t=args.t, a0=args.a0, lam=args.lam, mu=args.mu,
        temperature=args.temperature,
        **{name: getattr(args, name) for name in
           ("nk", "nseg", "bands", "forms", "pp_normalization", "units")},
        normalize_forms=not args.raw_forms,
    )
    p.validate()
    print(json.dumps(asdict(p), indent=2))
    print(f"Susceptibilities are reported in {susceptibility_unit(p)}.")
    results = run(out=args.out, parameters=p)
    print(f"Saved two figures, full matrices, diagonal CSVs, and parameters "
          f"to {results['out']}")
    import matplotlib.pyplot as plt
    if args.show:
        plt.show()
    elif not running_in_notebook():
        plt.close("all")
    return results


if __name__ == "__main__":
    main()
