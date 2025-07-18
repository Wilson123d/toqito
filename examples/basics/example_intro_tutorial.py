"""Introductory Tutorial
======================

This tutorial will illustrate the basics of how to use :code:`|toqito⟩`. This
will cover how to instantiate and use the fundamental objects that
:code:`|toqito⟩` provides; namely quantum states, channels, and measurements.
"""

# %%
# This is an introduction to the functionality in :code:`|toqito⟩` and is not meant to serve as an
# introduction to quantum information. For more information, please consult the book :footcite:`Chuang_2011_Quantum` or the freely available lecture notes :footcite:`Watrous_2018_TQI`.
#
# This tutorial assumes you have :code:`|toqito⟩` installed on your machine. If you
# do not, please consult the installation instructions in :ref:`getting_started_reference-label`.
#
# States
# ------
#
# A *quantum state* is a density operator
#
# .. math::
#    \rho \in \text{D}(\mathcal{X})
#
# where :math:`\mathcal{X}` is a complex Euclidean space and where
# :math:`\text{D}(\cdot)` represents the set of density matrices, that is, the
# set of matrices that are positive semidefinite with trace equal to :math:`1`.
#
# Quantum States
# ^^^^^^^^^^^^^^
#
# A complete overview of the scope of quantum states can be found
# `here <https://toqito.readthedocs.io/en/latest/autoapi/states/index.html>`_
#
# The standard basis ket vectors given as :math:`|0\rangle` and :math:`|1\rangle` where
#
# .. math::
#    | 0 \rangle = [1, 0]^{\text{T}} \quad \text{and} \quad | 1 \rangle = [0, 1]^{\text{T}}
#
# can be defined in :code:`|toqito⟩` as such

from toqito.matrices import standard_basis

# |0>
standard_basis(2)[0]

# %%
# To get the other ket

# |1>
standard_basis(2)[1]

# %%
# One may define one of the four Bell states written as
#
# .. math::
#    u_0 = \frac{1}{\sqrt{2}} \left(| 00 \rangle + | 11 \rangle \right)
#
# using :code:`|toqito⟩` as

import numpy as np

e_0, e_1 = standard_basis(2)
u_0 = 1 / np.sqrt(2) * (np.kron(e_0, e_0) + np.kron(e_1, e_1))
u_0


# %%
# The corresponding density operator of :math:`u_0` can be obtained from
#
# .. math::
#    \rho_0 = u_0 u_0^* = \frac{1}{2}
#    \begin{pmatrix}
#        1 & 0 & 0 & 1 \\
#        0 & 0 & 0 & 0 \\
#        0 & 0 & 0 & 0 \\
#        1 & 0 & 0 & 1
#    \end{pmatrix} \in \text{D}(\mathcal{X}).
#
# In :code:`|toqito⟩`, that can be obtained as

import numpy as np

e_0, e_1 = standard_basis(2)
u_0 = 1 / np.sqrt(2) * (np.kron(e_0, e_0) + np.kron(e_1, e_1))
rho_0 = u_0 @ u_0.conj().T
rho_0


# %%
# Alternatively, we may leverage the :code:`bell` function in :code:`|toqito⟩` to
# generate all four Bell states defined as
#
# .. math::
#    \begin{equation}
#        \begin{aligned}
#            u_0 = \frac{1}{\sqrt{2}} \left(| 00 \rangle + | 11 \rangle \right), &\quad
#            u_1 = \frac{1}{\sqrt{2}} \left(| 00 \rangle - | 11 \rangle \right), \\
#            u_2 = \frac{1}{\sqrt{2}} \left(| 01 \rangle + | 10 \rangle \right), &\quad
#            u_3 = \frac{1}{\sqrt{2}} \left(| 01 \rangle - | 10 \rangle \right),
#        \end{aligned}
#    \end{equation}
#
# in a more concise manner as

from toqito.states import bell
import numpy as np

bell(0)

# %%
# The Bell states constitute one such well-known class of quantum states. There
# are many other classes of states that are widely used in the field of quantum
# information. For instance, the GHZ state
#
# .. math::
#    | GHZ \rangle = \frac{1}{\sqrt{2}} \left( | 000 \rangle + | 111 \rangle \right)
#
# is a well-known 3-qubit quantum state. We can invoke this using :code:`|toqito⟩` as

from toqito.states import ghz

ghz(2, 3)


# %%
# While the 3-qubit form of the GHZ state is arguably the most notable, it is
# possible to define a generalized GHZ state
#
# .. math::
#    | GHZ_n \rangle = \frac{1}{\sqrt{n}} \left( | 0 \rangle^{\otimes n} + | 1
#    \rangle^{\otimes n} \right).
#
# This generalized state may be obtained in :code:`|toqito⟩` as well. For instance,
# here is the GHZ state :math:`\mathbb{C}^{4^{\otimes 7}}` as
#
# .. math::
#    \frac{1}{\sqrt{30}} \left(| 0000000 \rangle + 2| 1111111 \rangle + 3|
#    2222222 \rangle + 4| 3333333\rangle \right).

from toqito.states import ghz
import numpy as np

dim = 4
num_parties = 7
coeffs = [1 / np.sqrt(30), 2 / np.sqrt(30), 3 / np.sqrt(30), 4 / np.sqrt(30)]
vec = ghz(dim, num_parties, coeffs)
for idx in np.nonzero(vec)[0]:
    print(f"Index: {int(idx)}, Value: {vec[idx][0]:.8f}")


# %%
# Properties of Quantum States
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Given a quantum state, it is often useful to be able to determine certain
# *properties* of the state.
#
# For instance, we can check if a quantum state is pure, that is, if the density
# matrix that describes the state has rank 1.
#
# Any one of the Bell states serve as an example of a pure state

from toqito.states import bell
from toqito.state_props import is_pure

rho = bell(0) @ bell(0).conj().T
is_pure(rho)


# %%
# Another property that is useful is whether a given state is PPT (positive
# partial transpose), that is, whether the state remains positive after taking
# the partial transpose of the state.
#
# For quantum states consisting of shared systems of either dimension :math:`2
# \otimes 2` or :math:`2 \otimes 3`, the notion of whether a state is PPT serves
# as a method to determine whether a given quantum state is entangled or
# separable.
#
# As an example, any one of the Bell states constitute a canonical maximally
# entangled state over :math:`2 \otimes 2` and therefore should not satisfy the
# PPT criterion.

from toqito.states import bell
from toqito.state_props import is_ppt

rho = bell(2) @ bell(2).conj().T
is_ppt(rho)


# %%
# As we can see, the PPT criterion is :code:`False` for an entangled state in
# :math:`2 \otimes 2`.
#
# Determining whether a quantum state is separable or entangled is often useful
# but is, unfortunately, NP-hard. For a given density matrix represented by a
# quantum state, we can use :code:`|toqito⟩` to run a number of separability tests
# from the literature to determine if it is separable or entangled.
#
# For instance, the following bound-entangled tile state is found to be entangled
# (i.e. not separable).

import numpy as np
from toqito.state_props import is_separable
from toqito.states import tile

rho = np.identity(9)
for i in range(5):
    rho -= tile(i) @ tile(i).conj().T

rho /= 4
is_separable(rho)


# %%
# Further properties that one can check via :code:`|toqito⟩` may be found `on this page
# <https://toqito.readthedocs.io/en/latest/autoapi/state_props/index.html>`_.
#
# Distance Metrics for Quantum States
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Given two quantum states, it is often useful to have some way in which to
# quantify how similar or different one state is from another.
#
# One well known metric is the *fidelity* function defined for two quantum
# states. For two states :math:`\rho` and :math:`\sigma`, one defines the
# fidelity between :math:`\rho` and :math:`\sigma` as
#
# .. math::
#    || \sqrt{\rho} \sqrt{\sigma} ||_1,
#
# where :math:`|| \cdot ||_1` denotes the trace norm.
#
# The fidelity function yields a value between :math:`0` and :math:`1`, with
# :math:`0` representing the scenario where :math:`\rho` and :math:`\sigma` are
# as different as can be and where a value of :math:`1` indicates a scenario
# where :math:`\rho` and :math:`\sigma` are identical.
#
# Let us consider an example in :code:`|toqito⟩` where we wish to calculate the
# fidelity function between quantum states that happen to be identical.

from toqito.states import bell
from toqito.state_metrics import fidelity
import numpy as np

# Define two identical density operators.
rho = bell(0) @ bell(0).conj().T
sigma = bell(0) @ bell(0).conj().T

# Calculate the fidelity between `rho` and `sigma`
np.around(fidelity(rho, sigma), decimals=2)

# %%
# There are a number of other metrics one can compute on two density matrices
# including the trace norm, trace distance. These and others are also available
# in :code:`|toqito⟩`. For a full list of distance metrics one can compute on
# quantum states, consult the docs.
#
# Channels
# --------
#
# A *quantum channel* can be defined as a completely positive and trace
# preserving linear map.
#
# More formally, let :math:`\mathcal{X}` and :math:`\mathcal{Y}` represent
# complex Euclidean spaces and let :math:`\text{L}(\cdot)` represent the set of
# linear operators. Then a quantum channel, :math:`\Phi` is defined as
#
# .. math::
#    \Phi: \text{L}(\mathcal{X}) \rightarrow \text{L}(\mathcal{Y})
#
# such that :math:`\Phi` is completely positive and trace preserving.
#
# Quantum Channels
# ^^^^^^^^^^^^^^^^
#
# The partial trace operation is an often used in various applications of quantum
# information. The partial trace is defined as
#
#    .. math::
#        \left( \text{Tr} \otimes \mathbb{I}_{\mathcal{Y}} \right)
#        \left(X \otimes Y \right) = \text{Tr}(X)Y
#
# where :math:`X \in \text{L}(\mathcal{X})` and :math:`Y \in
# \text{L}(\mathcal{Y})` are linear operators over complex Euclidean spaces
# :math:`\mathcal{X}` and :math:`\mathcal{Y}`.
#
# Consider the following matrix
#
# .. math::
#    X = \begin{pmatrix}
#            1 & 2 & 3 & 4 \\
#            5 & 6 & 7 & 8 \\
#            9 & 10 & 11 & 12 \\
#            13 & 14 & 15 & 16
#        \end{pmatrix}.
#
# Taking the partial trace over the second subsystem of :math:`X` yields the following matrix
#
# .. math::
#    \text{Tr}_B(X) = \begin{pmatrix}
#                7 & 11 \\
#                23 & 27
#                \end{pmatrix}.
#
# By default, the partial trace function in :code:`|toqito⟩` takes the trace of the second
# subsystem.

from toqito.matrix_ops import partial_trace
import numpy as np

test_input_mat = np.arange(1, 17).reshape(4, 4)
partial_trace(test_input_mat)


# %%
# By specifying the :code:`sys = [0]` argument, we can perform the partial trace over the first
# subsystem (instead of the default second subsystem as done above). Performing the partial
# trace over the first subsystem yields the following matrix
#
# .. math::
#    X_{pt, 1} = \begin{pmatrix}
#                    12 & 14 \\
#                    20 & 22
#                \end{pmatrix}.

from toqito.matrix_ops import partial_trace
import numpy as np

test_input_mat = np.arange(1, 17).reshape(4, 4)
partial_trace(test_input_mat, sys=[0])


# %%
# Another often useful channel is the *partial transpose*. The *partial transpose*
# is defined as
#
#    .. math::
#        \left( \text{T} \otimes \mathbb{I}_{\mathcal{Y}} \right)
#        \left(X\right)
#
# where :math:`X \in \text{L}(\mathcal{X})` is a linear operator over the complex
# Euclidean space :math:`\mathcal{X}` and where :math:`\text{T}` is the transpose
# mapping :math:`\text{T} \in \text{T}(\mathcal{X})` defined as
#
# .. math::
#    \text{T}(X) = X^{\text{T}}
#
# for all :math:`X \in \text{L}(\mathcal{X})`.
#
# Consider the following matrix
#
# .. math::
#    X = \begin{pmatrix}
#            1 & 2 & 3 & 4 \\
#            5 & 6 & 7 & 8 \\
#            9 & 10 & 11 & 12 \\
#            13 & 14 & 15 & 16
#        \end{pmatrix}.
#
# Performing the partial transpose on the matrix :math:`X` over the second
# subsystem yields the following matrix
#
# .. math::
#    X^{T_B} = \begin{pmatrix}
#                1 & 5 & 3 & 7 \\
#                2 & 6 & 4 & 8 \\
#                9 & 13 & 11 & 15 \\
#                10 & 14 & 12 & 16
#                \end{pmatrix}.
#
# By default, in :code:`|toqito⟩`, the partial transpose function performs the transposition on
# the second subsystem as follows.

from toqito.matrix_ops import partial_transpose
import numpy as np

test_input_mat = np.arange(1, 17).reshape(4, 4)
partial_transpose(test_input_mat)


# %%
# By specifying the :code:`sys = [0]` argument, we can perform the partial transpose over the
# first subsystem (instead of the default second subsystem as done above). Performing the
# partial transpose over the first subsystem yields the following matrix
#
# .. math::
#    X_{pt, 1} = \begin{pmatrix}
#                    1 & 2 & 9 & 10 \\
#                    5 & 6 & 13 & 14 \\
#                    3 & 4 & 11 & 12 \\
#                    7 & 8 & 15 & 16
#                \end{pmatrix}.


from toqito.matrix_ops import partial_transpose
import numpy as np

test_input_mat = np.arange(1, 17).reshape(4, 4)
partial_transpose(test_input_mat, sys=[0])

# %%
# **Applying Quantum Channels**
#
# Another important operation when working with quantum channels is applying them to quantum states. :func:`.apply_channel` in :code:`|toqito⟩` provides a convenient way to apply a quantum channel (represented by its Choi matrix) to a given quantum state.
#
# Here, we illustrate how to apply two widely used channels – the depolarizing channel and the dephasing channel – using :func:`.apply_channel`.
#
# **Depolarizing Channel**
#
# The depolarizing channel replaces a state with the maximally mixed state with probability :math:`p` and leaves it unchanged with probability :math:`(1-p)`. Mathematically, it is defined as
#
# .. math::
#    \mathcal{N}(\rho) = (1-p) \rho + p\,\frac{\mathbb{I}}{d},
#
# where :math:`\mathbb{I}` is the identity operator and :math:`d` is the dimension of the Hilbert space. The example below applies the depolarizing channel with :math:`p=0.3` to the computational basis state :math:`|0\rangle`.

import numpy as np
from toqito.channel_ops import apply_channel
from toqito.channels import depolarizing

# Create a quantum state |0⟩⟨0|.
rho = np.array([[1, 0], [0, 0]])

# Generate the depolarizing channel Choi matrix with noise probability p = 0.3.
choi = depolarizing(2, 0.3)

# Apply the depolarizing channel using apply_channel.
output_state = apply_channel(rho, choi)
print(output_state)

# %%
# **Dephasing Channel**
#
# The dephasing channel reduces the off-diagonal elements of a density matrix without changing the diagonal entries, thereby diminishing quantum coherence. It is commonly expressed as
#
# .. math::
#    \mathcal{N}(\rho) = (1-p) \rho + p\, Z \rho Z,
#
# where :math:`Z` is the Pauli-Z operator and :math:`p` represents the dephasing probability. The example below demonstrates how to apply the dephasing channel with :math:`p=0.4` to the plus state :math:`|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)`.

import numpy as np
from toqito.channel_ops import apply_channel
from toqito.channels import dephasing

# Create a quantum state |+⟩⟨+|.
rho = np.array([[0.5, 0.5], [0.5, 0.5]])

# Generate the dephasing channel Choi matrix with dephasing probability p = 0.4.
choi = dephasing(2, 0.4)

# Apply the dephasing channel using apply_channel.
output_state = apply_channel(rho, choi)
print(output_state)


# %%
# Noisy Channels
# ^^^^^^^^^^^^^^
#
# Quantum noise channels model the interaction between quantum systems and their environment, resulting in decoherence and loss of quantum information. The :code:`|toqito⟩` library provides implementations of common noise models used in quantum information processing.
#
# **Phase Damping Channel**
#
# The phase damping channel models quantum decoherence where phase information is lost without any energy dissipation. It is characterized by a parameter :math:`\gamma` representing the probability of phase decoherence.
#
# .. math::
#    K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1 - \gamma} \end{pmatrix}, \quad
#    K_1 = \begin{pmatrix} 0 & 0 \\ 0 & \sqrt{\gamma} \end{pmatrix}
#
# The phase damping channel can be applied to a quantum state as follows:

from toqito.channels import phase_damping
import numpy as np

# Create a density matrix with coherence.
rho = np.array([[1, 0.5], [0.5, 1]])

# Apply phase damping with γ = 0.2.
result = phase_damping(rho, gamma=0.2)
print(result)

# %%
# Note that the off-diagonal elements (coherences) are reduced by a factor of :math:`\sqrt{1-\gamma}`, while the diagonal elements (populations) remain unchanged.
#
# **Amplitude Damping Channel**
#
# The amplitude damping channel models energy dissipation from a quantum system to its environment, such as the spontaneous emission of a photon. It is parameterized by :math:`\gamma`, representing the probability of losing a quantum of energy.
#
# .. math::
#    K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1 - \gamma} \end{pmatrix}, \quad
#    K_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}
#
# Here's how to use the amplitude damping channel:

from toqito.channels import amplitude_damping
import numpy as np

# Create a quantum state.
rho = np.array([[0.5, 0.5], [0.5, 0.5]])

# Apply amplitude damping with γ = 0.3.
result = amplitude_damping(rho, gamma=0.3)
print(result)

# %%
# **Bit-Flip Channel**
#
# The bit-flip channel randomly flips the state of a qubit with probability :math:`p`, analogous to the classical bit-flip error in classical information theory.
#
# .. math::
#    K_0 = \sqrt{1 - p} \, I = \sqrt{1 - p} \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \quad
#    K_1 = \sqrt{p} \, X = \sqrt{p} \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}


from toqito.channels import bitflip
import numpy as np

# Create a quantum state |0⟩⟨0|.
rho = np.array([[1, 0], [0, 0]])

# Apply bit-flip with probability = 0.25.
result = bitflip(rho, prob=0.25)
print(result)

# %%
# Observe that the result is a mixed state with 75% probability of being in state :math:`|0\rangle` and 25% probability of being in state :math:`|1\rangle`, as expected for a bit flip error with probability :math:`p = 0.25`.
#
# **Pauli Channel**
#
# The Pauli channel is a quantum noise model that applies a probabilistic mixture of Pauli operators
# to a quantum state. It is defined by a probability vector :math:`(p_0, \ldots, p_{4^q - 1})`, where
# :math:`q` is the number of qubits, and :math:`P_i` are the Pauli operators acting on the system.
#
# .. math::
#    \Phi(\rho) = \sum_{i=0}^{4^q - 1} p_i P_i \rho P_i^\dagger.
#
# For example, when :math:`q = 1`, the Pauli operators are:
# :math:`P_0 = I`, :math:`P_1 = X`, :math:`P_2 = Y`, and :math:`P_3 = Z`. For multiple qubits,
# these operators are extended as tensor products.
#
# It is also worth noting that when
#
# * :math:`P_2 = 0`, and :math:`P_3 = 0`, :func:`.pauli_channel` is equivalent to a :func:`.bitflip` channel
#
# * :math:`P_1 = 0`, and :math:`P_2 = 0`, :func:`.pauli_channel` is equivalent to a Phase Flip channel
#
# * :math:`P_1 = 0`, and :math:`P_3 = 0`, :func:`.pauli_channel` is equivalent to a Bit and Phase Flip channel
#
# The Pauli channel can be used to apply noise to an input quantum state or generate a Choi matrix.


from toqito.channels import pauli_channel
import numpy as np

# Define probabilities for single-qubit Pauli operators.
probabilities = np.array([0.5, 0.2, 0.2, 0.1])

# Define an input density matrix.
rho = np.array([[1, 0], [0, 0]])

# Apply the Pauli channel.
_, result = pauli_channel(prob=probabilities, input_mat=rho)
print(result)

# %%
# Here, the probabilities correspond to applying the identity (:math:`I`), bit-flip (:math:`X`),
# phase-flip (:math:`Z`), and combined bit-phase flip (:math:`Y`) operators.
#
# Measurements
# ------------
#
# A *measurement* can be defined as a function
#
# .. math::
#    \mu: \Sigma \rightarrow \text{Pos}(\mathcal{X})
#
# satisfying
#
# .. math::
#    \sum_{a \in \Sigma} \mu(a) = \mathbb{I}_{\mathcal{X}}
#
# where :math:`\Sigma` represents a set of measurement outcomes and where
# :math:`\mu(a)` represents the measurement operator associated with outcome
# :math:`a \in \Sigma`.
#
# POVM
# ^^^^
#
# POVM (Positive Operator-Valued Measure) is a set of positive operators that sum up to the identity.
#
# Consider the following matrices:
#
# .. math::
#    M_0 =
#    \begin{pmatrix}
#        1 & 0 \\
#        0 & 0
#    \end{pmatrix}
#    \quad \text{and} \quad
#    M_1 =
#    \begin{pmatrix}
#        0 & 0 \\
#        0 & 1
#    \end{pmatrix}.
#
# Our function expects this set of operators to be a POVM because it checks if the operators
# sum up to the identity, ensuring that the measurement outcomes are properly normalized.

from toqito.measurement_props import is_povm
import numpy as np

meas_1 = np.array([[1, 0], [0, 0]])
meas_2 = np.array([[0, 0], [0, 1]])
meas = [meas_1, meas_2]
is_povm(meas)

# %%
# Random POVM
# ^^^^^^^^^^^
#
# We may also use :func:`.random_povm` to randomly generate a POVM, and can verify that a
# randomly generated set satisfies the criteria for being a POVM set.

from toqito.measurement_props import is_povm
from toqito.rand import random_povm
import numpy as np

dim, num_inputs, num_outputs = 2, 2, 2
measurements = random_povm(dim, num_inputs, num_outputs)
is_povm([measurements[:, :, 0, 0], measurements[:, :, 0, 1]])

# %%
# Alternatively, the following matrices do not constitute a POVM set.
#
# .. math::
#    M_0 =
#    \begin{pmatrix}
#        1 & 2 \\
#        3 & 4
#    \end{pmatrix}
#    \quad \text{and} \quad
#    M_1 =
#    \begin{pmatrix}
#        5 & 6 \\
#        7 & 8
#    \end{pmatrix},

from toqito.measurement_props import is_povm
import numpy as np

non_meas_1 = np.array([[1, 2], [3, 4]])
non_meas_2 = np.array([[5, 6], [7, 8]])
non_meas = [non_meas_1, non_meas_2]
is_povm(non_meas)

# %%
# Measurement Operators
# ^^^^^^^^^^^^^^^^^^^^^
#
# Consider the following state:
#
# .. math::
#    u = \frac{1}{\sqrt{3}} e_0 + \sqrt{\frac{2}{3}} e_1
#
# where we define :math:`u u^* = \rho \in \text{D}(\mathcal{X})` and :math:`e_0`
# and :math:`e_1` are the standard basis vectors.
#
# .. math::
#    e_0 = \begin{pmatrix} 1 \\ 0 \end{pmatrix} \quad \text{and} \quad e_1 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}
#
# The measurement operators are defined as shown below:
#
# .. math::
#    P_0 = e_0 e_0^* \quad \text{and} \quad P_1 = e_1 e_1^*.

from toqito.matrices import standard_basis
from toqito.measurement_ops import measure
import numpy as np

e_0, e_1 = standard_basis(2)

u = (1 / np.sqrt(3)) * e_0 + (np.sqrt(2 / 3)) * e_1
rho = u @ u.conj().T

proj_0 = e_0 @ e_0.conj().T
proj_1 = e_1 @ e_1.conj().T

# %%
# Then the probability of obtaining outcome :math:`0` is given by
#
# .. math::
#    \langle P_0, \rho \rangle = \frac{1}{3}.

measure(proj_0, rho)

# %%
# Similarly, the probability of obtaining outcome :math:`1` is given by
#
# .. math::
#    \langle P_1, \rho \rangle = \frac{2}{3}.

measure(proj_1, rho)

# %%
# Pretty Good Measurement
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# Consider "pretty good measurement" on the set of trine states.
#
# The pretty good measurement (PGM), also known as the "square root measurement" is a set of POVMs :math:`(G_1, \ldots, G_n)` defined as
#
# .. math::
#    G_i = P^{-1/2} \left(p_i \rho_i\right) P^{-1/2} \quad \text{where} \quad P = \sum_{i=1}^n p_i \rho_i.
#
# This measurement was initially defined in :footcite:`Hughston_1993_Complete` and has found applications in quantum state discrimination tasks.
# While not always optimal, the PGM provides a reasonable measurement strategy that can be computed efficiently.
#
# For example, consider the following trine states:
#
# .. math::
#    u_0 = |0\rangle, \quad
#    u_1 = -\frac{1}{2}\left(|0\rangle + \sqrt{3}|1\rangle\right), \quad \text{and} \quad
#    u_2 = -\frac{1}{2}\left(|0\rangle - \sqrt{3}|1\rangle\right).

from toqito.states import trine
from toqito.measurements import pretty_good_measurement

states = trine()
probs = [1 / 3, 1 / 3, 1 / 3]
pgm = pretty_good_measurement(states, probs)
pgm

# %%
# Pretty Bad Measurement
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Similarly, we can consider so-called "pretty bad measurement" (PBM) on the set of trine states :footcite:`McIrvin_2024_Pretty`.
#
# The pretty bad measurement (PBM) is a set of POVMs :math:`(B_1, \ldots, B_n)` defined as
#
# .. math::
#    B_i = \left(P + (n-1)p_i \rho_i\right)^{-1} p_i \rho_i \left(P + (n-1)p_i \rho_i\right)^{-1} \quad \text{where} \quad P = \sum_{i=1}^n p_i \rho_i.
#
# Like the PGM, the PBM provides a measurement strategy for quantum state discrimination, but with different properties that can be useful in certain contexts.
#
# .. math::
#    u_0 = |0\rangle, \quad
#    u_1 = -\frac{1}{2}\left(|0\rangle + \sqrt{3}|1\rangle\right), \quad \text{and} \quad
#    u_2 = -\frac{1}{2}\left(|0\rangle - \sqrt{3}|1\rangle\right).

from toqito.states import trine
from toqito.measurements import pretty_bad_measurement

states = trine()
probs = [1 / 3, 1 / 3, 1 / 3]
pbm = pretty_bad_measurement(states, probs)
pbm

# %%
#
#
# References
# ----------
#
# .. footbibliography::
