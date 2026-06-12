"""
Variational Quantum Circuit (VQC) layers for hybrid quantum-classical models.

Two classes are provided:

    QuantumLinear           — maps n_qubits → n_qubits, NO classical bottleneck.
                              Use when input dim already equals the qubit count by design
                              (experiments 02 and 03: embed_dim == n_qubits).

    QuantumLinearWithAdapter — maps n_in → n_qubits via a trainable linear layer,
                              then runs the same VQC. Use when the upstream embedding
                              is larger than n_qubits and must be compressed first
                              (experiment 01: n_embd >> n_qubits).

Both share the same quantum pipeline:

    input → tanh(x) · π  →  AngleEmbedding  →  StronglyEntanglingLayers  →  ⟨Z_i⟩  →  output

Weight shape of the variational circuit: (n_qlayers, n_qubits, 3).
Differentiation via diff_method="backprop" (torch interface, GPU-compatible).
"""

import torch
import torch.nn as nn
import pennylane as qml


class QuantumLinear(nn.Module):
    """
    A quantum linear layer that maps n_qubits → n_qubits.

    There is NO classical bottleneck: the caller is responsible for ensuring
    that the input's last dimension equals n_qubits. In experiments 02 and 03
    this is guaranteed by the constraint embed_dim == n_qubits.

    Pipeline:
        input (n_qubits,) → tanh · π → AngleEmbedding → StronglyEntanglingLayers → ⟨Z⟩ → output (n_qubits,)

    Args:
        n_qubits  (int): Number of qubits. Also the input and output dimension.
        n_qlayers (int): Depth of StronglyEntanglingLayers.
        q_device  (str): PennyLane device string, e.g. "default.qubit" or "lightning.qubit".
    """

    def __init__(self, n_qubits: int, n_qlayers: int, q_device: str = "default.qubit"):
        super().__init__()
        self.n_qubits = n_qubits

        dev = qml.device(q_device, wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_qlayers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., n_qubits) — any leading batch/sequence dimensions are supported.
        Returns:
            Tensor of same shape as x. Each value is a Pauli-Z expectation in [-1, 1].
        """
        leading = x.shape[:-1]
        x_flat = x.reshape(-1, self.n_qubits)

        # Scale to [-π, π]: tanh keeps gradients alive, π fills the Bloch sphere.
        x_scaled = torch.tanh(x_flat) * torch.pi

        out = self.qlayer(x_scaled)
        return out.reshape(*leading, self.n_qubits)


class QuantumLinearWithAdapter(nn.Module):
    """
    A quantum linear layer with a classical compression adapter.

    Adds a trainable nn.Linear(n_in, n_qubits) before the VQC. This solves
    the dimensionality mismatch when the upstream embedding is larger than
    the number of simulatable qubits (experiment 01: n_embd=32, n_qubits=4).

    Pipeline:
        input (n_in,) → Linear(n_in, n_qubits) → tanh · π → VQC → ⟨Z⟩ → output (n_qubits,)

    Args:
        n_in      (int): Input dimension (e.g., embedding size).
        n_qubits  (int): Number of qubits and output dimension.
        n_qlayers (int): Depth of StronglyEntanglingLayers.
        q_device  (str): PennyLane device string.
    """

    def __init__(
        self,
        n_in: int,
        n_qubits: int,
        n_qlayers: int,
        q_device: str = "default.qubit",
    ):
        super().__init__()
        self.n_in = n_in
        self.n_qubits = n_qubits

        # Classical adapter: compresses high-dimensional input to qubit count.
        self.adapter = nn.Linear(n_in, n_qubits)

        # Core VQC reuses QuantumLinear to avoid duplication.
        self.vqc = QuantumLinear(n_qubits, n_qlayers, q_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., n_in) — any leading batch/sequence dimensions are supported.
        Returns:
            (..., n_qubits) — output of the VQC.
        """
        leading = x.shape[:-1]
        x_flat = x.reshape(-1, self.n_in)
        x_compressed = self.adapter(x_flat)
        out = self.vqc(x_compressed.reshape(*leading, self.n_qubits))
        return out
