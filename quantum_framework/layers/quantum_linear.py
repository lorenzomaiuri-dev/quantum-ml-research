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


def simulator_device(q_device: str) -> torch.device:
    """
    Torch device on which a PennyLane device's tensors must live.

    The statevector simulators (`default.qubit`, `lightning.qubit`) build their
    initial state on CPU, so feeding them CUDA tensors raises a device mismatch
    inside the first gate application. Only the `.gpu` backends own CUDA memory.
    """
    return torch.device("cuda" if q_device.endswith(".gpu") else "cpu")


class PinnedTorchLayer(qml.qnn.TorchLayer):
    """
    A TorchLayer whose weights stay on the simulator's device.

    A hybrid model is normally sent to the GPU as a whole, but the variational
    weights must remain where the simulator can use them, so they are moved back
    after every `.to()` / `.cuda()` applied to an enclosing module.
    """

    def __init__(self, qnode, weight_shapes: dict, sim_device: torch.device):
        super().__init__(qnode, weight_shapes)
        self.sim_device = sim_device
        self._pin()

    def _pin(self):
        for tensor in list(self.parameters(recurse=True)) + list(
            self.buffers(recurse=True)
        ):
            if tensor.device != self.sim_device:
                tensor.data = tensor.data.to(self.sim_device)
                if getattr(tensor, "grad", None) is not None:
                    tensor.grad.data = tensor.grad.data.to(self.sim_device)

    def _apply(self, fn, *args, **kwargs):
        super()._apply(fn, *args, **kwargs)
        self._pin()
        return self

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Evaluate the circuit on the simulator's device, answer on the caller's.

        `.to()` is autograd-aware, so gradients flow back to the classical part
        of the model unchanged.
        """
        out = super().forward(inputs.to(self.sim_device))
        return out.to(device=inputs.device, dtype=inputs.dtype)


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
        self.qlayer = PinnedTorchLayer(
            circuit, weight_shapes, simulator_device(q_device)
        )

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
