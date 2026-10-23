#!/usr/bin/env python3
"""Six-band q=0 kagome 120-degree antiferromagnet.

Basis: (1 up, 1 down, 2 up, 2 down, 3 up, 3 down).
Energies are in eV, lengths in Angstrom, and k in inverse Angstrom.
"""

import numpy as np


# Model parameters
t = 1                 # nearest-neighbor hopping (eV)
J = 1.7                # exchange coupling (eV)
lam = 0.2                # SOC strength lambda (eV)
a = 5.0                  # Bravais-vector length (Angstrom)
E_F = 0.0                # Fermi energy (eV)
kBT = 0.010              # Fermi-Dirac width (eV)
Nk = 401                 # Nk x Nk BZ mesh
kappa = -1               # chirality: +1 or -1
delta_phi = -90.0        # common rotation DeltaPhi (degrees)
theta = 90.0              # polar angle of all moments (degrees)


# Magnetic texture: Phi_i = kappa*phi_i + DeltaPhi.
phi = np.array([-150.0, -30.0, 90.0])


# Directed SOC vectors after n -> -n
n12 = np.array([0.0, 1.0, 0.0])
n23 = np.array([-np.sqrt(3.0) / 2.0, -0.5, 0.0])
n31 = np.array([np.sqrt(3.0) / 2.0, -0.5, 0.0])


# Spin matrices
sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
sy = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
s0 = np.eye(2, dtype=complex)
orbital_identity = np.eye(3, dtype=complex)

SPIN_OPERATORS = {
    "1": np.kron(orbital_identity, sx),
    "2": np.kron(orbital_identity, sy),
    "3": np.kron(orbital_identity, sz),
}


class KagomeModel:
    """Callable Kagome Hamiltonian with independent model parameters."""

    def __init__(
        self,
        t: float = t,
        J: float = J,
        lam: float = lam,
        a: float = a,
        E_F: float = E_F,
        kBT: float = kBT,
        kappa: int = kappa,
        delta_phi: float = delta_phi,
        theta: float = theta,
    ) -> None:
        if kappa not in (-1, 1):
            raise ValueError("kappa must be +1 or -1")

        self.t = t
        self.J = J
        self.lam = lam
        self.a = a
        self.E_F = E_F
        self.kBT = kBT
        self.kappa = kappa
        self.delta_phi = delta_phi
        self.theta = theta

        self.a1 = a * np.array([1.0, 0.0])
        self.a2 = a * np.array([0.5, np.sqrt(3.0) / 2.0])
        self.b1 = (2.0 * np.pi / a) * np.array(
            [1.0, -1.0 / np.sqrt(3.0)]
        )
        self.b2 = (2.0 * np.pi / a) * np.array(
            [0.0, 2.0 / np.sqrt(3.0)]
        )

        self.d12 = self.a1 / 2.0
        self.d23 = (self.a2 - self.a1) / 2.0
        self.d31 = -self.a2 / 2.0  
        self.phi = phi.copy()
        self.Phi = self.kappa * self.phi + self.delta_phi
        self.n12 = n12.copy()
        self.n23 = n23.copy()
        self.n31 = n31.copy()
        self.SPIN_OPERATORS = SPIN_OPERATORS

        moments = np.column_stack(
            (
                np.cos(np.deg2rad(self.Phi)) * np.sin(np.deg2rad(self.theta)),
                np.sin(np.deg2rad(self.Phi)) * np.sin(np.deg2rad(self.theta)),
                np.full(3, np.cos(np.deg2rad(self.theta))),
            )
        )
        self.H_exchange = np.zeros((6, 6), dtype=complex)
        for site in range(3):
            block = slice(2 * site, 2 * site + 2)
            self.H_exchange[block, block] = -self.J * self.sigma(moments[site])

    @staticmethod
    def sigma(vector: np.ndarray) -> np.ndarray:
        return vector[0] * sx + vector[1] * sy + vector[2] * sz

    def h_hop(self, kx: float, ky: float) -> np.ndarray:
        k = np.array([kx, ky])
        A = 2.0 * np.cos(k @ self.d12)
        B = 2.0 * np.cos(k @ self.d23)
        C = 2.0 * np.cos(k @ self.d31)

        h = np.zeros((6, 6), dtype=complex)
        h[0:2, 2:4] = h[2:4, 0:2] = self.t * A * s0
        h[2:4, 4:6] = h[4:6, 2:4] = self.t * B * s0
        h[4:6, 0:2] = h[0:2, 4:6] = self.t * C * s0
        return h

    def h_soc(self, kx: float, ky: float) -> np.ndarray:
        k = np.array([kx, ky])
        A = 2.0 * np.cos(k @ self.d12)
        B = 2.0 * np.cos(k @ self.d23)
        C = 2.0 * np.cos(k @ self.d31)

        T12 = 1.0j * self.lam * A * self.sigma(self.n12)
        T23 = 1.0j * self.lam * B * self.sigma(self.n23)
        T31 = 1.0j * self.lam * C * self.sigma(self.n31)

        h = np.zeros((6, 6), dtype=complex)
        h[0:2, 2:4], h[2:4, 0:2] = T12, T12.conj().T
        h[2:4, 4:6], h[4:6, 2:4] = T23, T23.conj().T
        h[4:6, 0:2], h[0:2, 4:6] = T31, T31.conj().T
        return h

    def hamiltonian(self, kx: float, ky: float) -> np.ndarray:
        return self.h_hop(kx, ky) + self.H_exchange + self.h_soc(kx, ky)

    def dh_dk(self, direction: str, kx: float, ky: float) -> np.ndarray:
        if direction not in "12":
            raise ValueError("direction must be '1' (x) or '2' (y)")
        axis = int(direction) - 1
        k = np.array([kx, ky])

        A_c = -2.0 * self.d12[axis] * np.sin(k @ self.d12)
        B_c = -2.0 * self.d23[axis] * np.sin(k @ self.d23)
        C_c = -2.0 * self.d31[axis] * np.sin(k @ self.d31)
        T12_c = A_c * (self.t * s0 + 1.0j * self.lam * self.sigma(self.n12))
        T23_c = B_c * (self.t * s0 + 1.0j * self.lam * self.sigma(self.n23))
        T31_c = C_c * (self.t * s0 + 1.0j * self.lam * self.sigma(self.n31))

        derivative = np.zeros((6, 6), dtype=complex)
        derivative[0:2, 2:4], derivative[2:4, 0:2] = T12_c, T12_c.conj().T
        derivative[2:4, 4:6], derivative[4:6, 2:4] = T23_c, T23_c.conj().T
        derivative[4:6, 0:2], derivative[0:2, 4:6] = T31_c, T31_c.conj().T
        return derivative

    def d2h_dk2(
        self,
        direction_b: str,
        direction_c: str,
        kx: float,
        ky: float,
    ) -> np.ndarray:
        if direction_b not in "12" or direction_c not in "12":
            raise ValueError("directions must be '1' or '2'")
        b, c = int(direction_b) - 1, int(direction_c) - 1
        k = np.array([kx, ky])

        A_bc = -2.0 * self.d12[b] * self.d12[c] * np.cos(k @ self.d12)
        B_bc = -2.0 * self.d23[b] * self.d23[c] * np.cos(k @ self.d23)
        C_bc = -2.0 * self.d31[b] * self.d31[c] * np.cos(k @ self.d31)
        T12_bc = A_bc * (self.t * s0 + 1.0j * self.lam * self.sigma(self.n12))
        T23_bc = B_bc * (self.t * s0 + 1.0j * self.lam * self.sigma(self.n23))
        T31_bc = C_bc * (self.t * s0 + 1.0j * self.lam * self.sigma(self.n31))

        derivative = np.zeros((6, 6), dtype=complex)
        derivative[0:2, 2:4], derivative[2:4, 0:2] = T12_bc, T12_bc.conj().T
        derivative[2:4, 4:6], derivative[4:6, 2:4] = T23_bc, T23_bc.conj().T
        derivative[4:6, 0:2], derivative[0:2, 4:6] = T31_bc, T31_bc.conj().T
        return derivative
