"""Bare susceptibilities of the Lieb lattice, written out step by step.

Same physics as lieb_susceptibility.py, with the projector algebra, the
channel matrices and the command line stripped out: diagonalize the 3x3
Bloch Hamiltonian with eigh, sum the two textbook bubbles over the
Brillouin zone, plot the result.

  particle-hole   chi_ss'(q) = (1/A) sum_k sum_nm |<u_m^s'(k-q/2)|u_n^s(k+q/2)>|^2
                               * [f(xi_m) - f(xi_n)] / (xi_n - xi_m)

  particle-particle  Pi_ss'(Q) = (1/A) sum_k sum_nm |<u_m^s'(-k+Q/2)*|u_n^s(k+Q/2)>|^2
                               * [1 - f(xi_n) - f(xi_m)] / (xi_n + xi_m)

The pairing partner is the time-reversed state, so the second bra in Pi
is conjugated once more than in chi -- that single conjugate is the only
formal difference between the two bubbles.

Energies in eV, lengths in Angstrom, susceptibilities in 1/(eV A^2).
Edit the constants below and run: python3 lieb_simple.py
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Parameters. Values for the CuO2-plane realization of the Lieb lattice.
# ---------------------------------------------------------------------
T = 1.3           # Nearest-neighbour hopping, eV.
A0 = 3.8          # Lattice constant, Angstrom.
LAM = 0.20 * T    # Intrinsic spin-orbit coupling, eV.
MU = 1.20 * T     # Chemical potential from the flat band, eV.
KT = 0.04 * T     # Temperature k_B T, eV.

NK = 120          # NK x NK momentum grid over the Brillouin zone.
NSEG = 60         # Points per segment of the Gamma-X-M-Gamma path.
UPPER_BAND_ONLY = True   # False also sums the flat and lower bands.

UP, DOWN = +1, -1


# ---------------------------------------------------------------------
# 1. Bloch Hamiltonian and its eigenstates
# ---------------------------------------------------------------------
def hamiltonian(kx, ky, spin):
    """The 3x3 Lieb Hamiltonian in the (B, A, C) orbital basis.

    A and B, and A and C, are connected by the hopping; the spin-orbit
    term connects B and C with a sign set by the spin.
    """
    ax = -2 * T * np.cos(A0 * kx / 2)
    ay = -2 * T * np.cos(A0 * ky / 2)
    az = 4 * LAM * np.sin(A0 * kx / 2) * np.sin(A0 * ky / 2)
    h = np.zeros((len(kx), 3, 3), dtype=complex)
    h[:, 0, 1] = h[:, 1, 0] = ax
    h[:, 1, 2] = h[:, 2, 1] = ay
    h[:, 0, 2] = -1j * spin * az
    h[:, 2, 0] = +1j * spin * az
    return h


def eigenstates(kx, ky, spin):
    """Band energies measured from mu, and the eigenvectors.

    Returns xi of shape (k, band) and u of shape (k, orbital, band).
    The three bands come out sorted as -a, 0, +a, so the last one is
    the upper band.
    """
    energy, vector = np.linalg.eigh(hamiltonian(kx, ky, spin))
    if UPPER_BAND_ONLY:
        energy, vector = energy[:, 2:], vector[:, :, 2:]
    return energy - MU, vector


# ---------------------------------------------------------------------
# 2. The two static kernels
# ---------------------------------------------------------------------
def minus_fermi_derivative(x):
    """-f'(x), written with exp(-|x|) so it never overflows."""
    z = np.exp(-np.abs(x) / KT)
    return z / (KT * (1 + z)**2)


def lindhard_kernel(x, y):
    """[f(y) - f(x)] / (x - y), equal to -f'(x) when x = y."""
    difference = x - y
    coincident = np.abs(difference) < 1e-9
    safe = np.where(coincident, 1.0, difference)
    value = 0.5 * (np.tanh(x / (2 * KT)) - np.tanh(y / (2 * KT))) / safe
    return np.where(coincident, minus_fermi_derivative(x), value)


def cooper_kernel(x, y):
    """[1 - f(x) - f(y)] / (x + y). It is the Lindhard kernel at -y."""
    return lindhard_kernel(x, -y)


# ---------------------------------------------------------------------
# 3. Brillouin-zone grid and the s- and p-wave weights
# ---------------------------------------------------------------------
def brillouin_zone():
    """NK x NK midpoints of [-pi/a0, pi/a0)^2, flattened."""
    axis = (2 * np.pi / A0) * ((np.arange(NK) + 0.5) / NK - 0.5)
    kx, ky = np.meshgrid(axis, axis, indexing="ij")
    return kx.ravel(), ky.ravel()


def form_weights(kx, ky):
    """|f(k)|^2 for each pairing/ordering symmetry, averaging to one.

    f_s = 1, f_px = sqrt(2) sin(a0 kx), f_py = sqrt(2) sin(a0 ky).
    """
    return {
        "s": np.ones_like(kx),
        "px": 2 * np.sin(A0 * kx)**2,
        "py": 2 * np.sin(A0 * ky)**2,
    }


KX, KY = brillouin_zone()
WEIGHTS = form_weights(KX, KY)


# ---------------------------------------------------------------------
# 4. The bubbles themselves
# ---------------------------------------------------------------------
def bubble(qx, qy, kind, spin1, spin2):
    """One bubble at one external momentum, for each symmetry weight.

    kind is "PH" for chi(q) or "PP" for Pi(Q). Returns a dictionary
    keyed like WEIGHTS, in 1/(eV Angstrom^2).
    """
    xi1, u1 = eigenstates(KX + qx / 2, KY + qy / 2, spin1)
    if kind == "PH":
        xi2, u2 = eigenstates(KX - qx / 2, KY - qy / 2, spin2)
        # <u_m(k-q/2)|u_n(k+q/2)>, the usual coherence factor.
        overlap = np.einsum("kan,kam->knm", u1, u2.conj())
        kernel = lindhard_kernel(xi1[:, :, None], xi2[:, None, :])
    else:
        xi2, u2 = eigenstates(-KX + qx / 2, -KY + qy / 2, spin2)
        # The partner is time reversed, hence u2 and not u2.conj().
        overlap = np.einsum("kan,kam->knm", u1, u2)
        kernel = cooper_kernel(xi1[:, :, None], xi2[:, None, :])

    # Sum over both band indices; what is left depends on k alone.
    summand = (np.abs(overlap)**2 * kernel).sum(axis=(1, 2))
    # (1/A) sum_k  ->  mean over the grid, divided by the cell area.
    return {name: np.mean(w * summand) / A0**2 for name, w in WEIGHTS.items()}


def path():
    """Gamma-X-M-Gamma, with the distance travelled and the corners."""
    corners = (np.pi / A0) * np.array([[0, 0], [1, 0], [1, 1], [0, 0]], float)
    points = np.concatenate(
        [np.linspace(corners[i], corners[i + 1], NSEG, endpoint=False)
         for i in range(3)] + [corners[-1:]]
    )
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1) * A0
    distance = np.concatenate([[0.0], np.cumsum(steps)])
    return points, distance, distance[np.arange(4) * NSEG]


def sweep(kind, spin2):
    """Walk the path, collecting one curve per symmetry weight."""
    points, distance, corners = path()
    curves = {name: np.empty(len(points)) for name in WEIGHTS}
    for i, (qx, qy) in enumerate(points):
        values = bubble(qx, qy, kind, UP, spin2)
        for name in curves:
            curves[name][i] = values[name]
        if i % 20 == 0:
            print(f"  {kind} spin2={spin2:+d}: {i + 1}/{len(points)}", flush=True)
    return distance, corners, curves


# ---------------------------------------------------------------------
# 5. Plots. Colour carries the symmetry, dashes carry the spin pair.
# ---------------------------------------------------------------------
COLORS = {"s": "#2a78d6", "px": "#eb6834", "py": "#4a3aa7"}
LABELS = {"s": "$s$", "px": "$p_x$", "py": "$p_y$"}


def draw(axes, distance, curves, forms, style, spin_label):
    for name in forms:
        axes.plot(distance, curves[name], color=COLORS[name], ls=style,
                  lw=2.0, label=f"{spin_label}, {LABELS[name]}")


def finish(axes, corners, title, ylabel, xlabel):
    for corner in corners[1:-1]:
        axes.axvline(corner, color="#d8d8d5", lw=0.8, zorder=0)
    axes.set_xticks(corners, [r"$\Gamma$", "$X$", "$M$", r"$\Gamma$"])
    axes.set_xlim(corners[0], corners[-1])
    axes.set_ylim(bottom=0)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title, loc="left", weight="bold", pad=10)
    axes.grid(axis="y", color="#ececea", lw=0.7)
    for edge in ("top", "right"):
        axes.spines[edge].set_visible(False)
    axes.legend(loc="upper left", bbox_to_anchor=(1.06, 1), frameon=False,
                borderaxespad=0)


def main():
    print(f"Lieb lattice: t={T} eV, a0={A0} A, lambda={LAM:.3g} eV, "
          f"mu={MU:.3g} eV, kT={KT:.3g} eV, {NK}x{NK} grid")
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 13,
                         "savefig.dpi": 180})

    # Particle-hole: equal spins (spin2 = up) and opposite (spin2 = down).
    distance, corners, equal = sweep("PH", UP)
    _, _, opposite = sweep("PH", DOWN)
    figure, axes = plt.subplots(figsize=(9.5, 5.4), layout="constrained")
    draw(axes, distance, equal, ("s", "px", "py"), "-", "equal spin")
    draw(axes, distance, opposite, ("s", "px", "py"), "--", "opposite spin")
    finish(axes, corners, "Particle-hole susceptibility",
           r"$\chi(\mathbf{q})$  [$\mathrm{eV}^{-1}\,\mathrm{\AA}^{-2}$]",
           r"Transfer momentum $\mathbf{q}$")
    figure.savefig("chi_simple.png")

    # Particle-particle: opposite spins pair as the singlet and as T0,
    # equal spins as T+1 = T-1. Equal-spin s-wave is Pauli forbidden.
    distance, corners, singlet = sweep("PP", DOWN)
    _, _, triplet = sweep("PP", UP)
    figure, axes = plt.subplots(figsize=(9.5, 5.4), layout="constrained")
    draw(axes, distance, singlet, ("s", "px", "py"), "-", "$S$ / $T_0$")
    draw(axes, distance, triplet, ("px", "py"), "--", r"$T_{\pm 1}$")
    finish(axes, corners, "Particle-particle susceptibility",
           r"$\Pi(\mathbf{Q})$  [$\mathrm{eV}^{-1}\,\mathrm{\AA}^{-2}$]",
           r"Pair momentum $\mathbf{Q}$")
    figure.savefig("pi_simple.png")

    print("Wrote chi_simple.png and pi_simple.png")
    return distance, corners, equal, opposite, singlet, triplet


if __name__ == "__main__":
    main()
