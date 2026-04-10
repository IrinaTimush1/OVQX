from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.utils import QuantumInstance

try:
    from qiskit import Aer
except Exception:
    from qiskit_aer import Aer


# =========================================================
# 1. DATA
# =========================================================

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "qnn-data.csv"

# Paper-trained parameters from Section 4.2
THETA_TRAINED = [3.860, -1.070, -1.583, 0.860]


def load_dataset():
    df = pd.read_csv(DATA_PATH, sep=";")
    X = df[["x0", "x1"]].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=int)
    return X, y


# =========================================================
# 2. BUILD THE QNN CIRCUIT (FIGURE 12)
# =========================================================

def build_qnn_circuit():
    qc = QuantumCircuit(2, 2)

    # Feature parameters
    feat_p0 = Parameter("feat_p0")
    feat_p1 = Parameter("feat_p1")
    feat_p2 = Parameter("feat_p2")
    feat_p3 = Parameter("feat_p3")
    feat_p4 = Parameter("feat_p4")
    feat_p5 = Parameter("feat_p5")

    # Trainable parameters
    theta_0 = Parameter("theta_0")
    theta_1 = Parameter("theta_1")
    theta_2 = Parameter("theta_2")
    theta_3 = Parameter("theta_3")

    # First feature-map block
    qc.h(0)
    qc.p(feat_p0, 0)
    qc.h(1)
    qc.p(feat_p1, 1)
    qc.cx(0, 1)
    qc.p(feat_p2, 1)
    qc.cx(0, 1)

    # Second feature-map block
    qc.h(0)
    qc.p(feat_p3, 0)
    qc.h(1)
    qc.p(feat_p4, 1)
    qc.cx(0, 1)
    qc.p(feat_p5, 1)
    qc.cx(0, 1)

    # Trainable layer
    qc.ry(theta_0, 0)
    qc.ry(theta_1, 1)
    qc.cx(0, 1)
    qc.ry(theta_2, 0)
    qc.ry(theta_3, 1)

    return qc


# =========================================================
# 3. BIND ONE DATAPOINT TO THE CIRCUIT
# =========================================================

def bind_qnn_circuit(qc, x0, x1, theta_values):
    phi1 = 2.0 * x0
    phi2 = 2.0 * x1
    phi_cross = 2.0 * (np.pi - x0) * (np.pi - x1)

    bind_dict = {
        "feat_p0": phi1,
        "feat_p1": phi2,
        "feat_p2": phi_cross,
        "feat_p3": phi1,
        "feat_p4": phi2,
        "feat_p5": phi_cross,
        "theta_0": theta_values[0],
        "theta_1": theta_values[1],
        "theta_2": theta_values[2],
        "theta_3": theta_values[3],
    }

    actual_bindings = {}
    for p in qc.parameters:
        actual_bindings[p] = bind_dict[p.name]

    return qc.assign_parameters(actual_bindings)


# =========================================================
# 4. READOUT SEMANTICS
# =========================================================

@dataclass
class ReadoutSemantics:
    name: str
    bit_order: str          # "qiskit_default" or "reversed"
    chosen_qubit: int       # 0 or 1
    invert_label: bool      # whether to flip 0<->1
    shots: int              # simulator shots
    decision_rule: str      # "single", "majority", "probability"


def extract_bit_from_bitstring(bitstring: str, chosen_qubit: int, bit_order: str) -> int:
    """
    Return the bit associated with chosen_qubit under a given convention.

    qiskit_default:
        bitstring is c[n-1]...c[0]
        so qubit 0 -> rightmost, qubit 1 -> leftmost for 2 qubits

    reversed:
        bitstring interpreted as c[0]...c[n-1]
        so qubit 0 -> leftmost, qubit 1 -> rightmost for 2 qubits
    """
    if len(bitstring) != 2:
        raise ValueError(f"Expected 2-bit measurement string, got {bitstring}")

    if bit_order == "qiskit_default":
        idx = -(chosen_qubit + 1)
        return int(bitstring[idx])

    if bit_order == "reversed":
        idx = chosen_qubit
        return int(bitstring[idx])

    raise ValueError(f"Unknown bit_order: {bit_order}")


def predict_from_counts(counts: Dict[str, int], semantics: ReadoutSemantics) -> int:
    """
    Predict class label from measurement counts under a given interpretation.
    """
    if semantics.decision_rule == "single":
        # shots should be 1, but if not, just take the most frequent bitstring
        bitstring = max(counts.items(), key=lambda kv: kv[1])[0]
        pred = extract_bit_from_bitstring(bitstring, semantics.chosen_qubit, semantics.bit_order)

    elif semantics.decision_rule == "majority":
        total = 0
        weighted_ones = 0
        for bitstring, c in counts.items():
            b = extract_bit_from_bitstring(bitstring, semantics.chosen_qubit, semantics.bit_order)
            weighted_ones += b * c
            total += c
        pred = 1 if weighted_ones >= total / 2 else 0

    elif semantics.decision_rule == "probability":
        total = 0
        weighted_ones = 0
        for bitstring, c in counts.items():
            b = extract_bit_from_bitstring(bitstring, semantics.chosen_qubit, semantics.bit_order)
            weighted_ones += b * c
            total += c
        p1 = weighted_ones / total
        pred = 1 if p1 >= 0.5 else 0

    else:
        raise ValueError(f"Unknown decision_rule: {semantics.decision_rule}")

    if semantics.invert_label:
        pred = 1 - pred

    return pred


# =========================================================
# 5. EVALUATION
# =========================================================

def evaluate_dataset(qc, X, y, theta_values, semantics: ReadoutSemantics, n_repeats: int = 50):
    backend = Aer.get_backend("qasm_simulator")
    qi = QuantumInstance(backend=backend, shots=semantics.shots)

    all_accuracies = []

    for _ in range(n_repeats):
        correct = 0

        for i in range(len(y)):
            x0, x1 = X[i]
            y_true = int(y[i])

            bound_qc = bind_qnn_circuit(qc, x0, x1, theta_values)
            qc_meas = bound_qc.copy()
            qc_meas.measure(range(2), range(2))

            result = qi.execute([qc_meas])
            counts = result.get_counts(qc_meas)

            y_pred = predict_from_counts(counts, semantics)

            if y_pred == y_true:
                correct += 1

        all_accuracies.append(correct / len(y))

    return {
        "name": semantics.name,
        "mean_acc": float(np.mean(all_accuracies)),
        "std_acc": float(np.std(all_accuracies)),
        "all_accs": all_accuracies,
        "semantics": semantics,
    }


# =========================================================
# 6. SEARCH PLAUSIBLE CONVENTIONS
# =========================================================

def build_candidate_semantics() -> List[ReadoutSemantics]:
    candidates = []

    # one-shot candidates: closest to the paper description
    for bit_order in ["qiskit_default", "reversed"]:
        for chosen_qubit in [0, 1]:
            for invert_label in [False, True]:
                candidates.append(
                    ReadoutSemantics(
                        name=f"one-shot | order={bit_order} | qubit={chosen_qubit} | invert={invert_label}",
                        bit_order=bit_order,
                        chosen_qubit=chosen_qubit,
                        invert_label=invert_label,
                        shots=1,
                        decision_rule="single",
                    )
                )

    # more stable majority-vote alternatives, in case paper's wording hides this
    for bit_order in ["qiskit_default", "reversed"]:
        for chosen_qubit in [0, 1]:
            for invert_label in [False, True]:
                candidates.append(
                    ReadoutSemantics(
                        name=f"majority-100 | order={bit_order} | qubit={chosen_qubit} | invert={invert_label}",
                        bit_order=bit_order,
                        chosen_qubit=chosen_qubit,
                        invert_label=invert_label,
                        shots=100,
                        decision_rule="majority",
                    )
                )

    return candidates


# =========================================================
# 7. MAIN
# =========================================================

def main():
    X, y = load_dataset()
    qc = build_qnn_circuit()

    print("=== INSTRUCTION ORDER ===")
    for idx, (instr, qargs, cargs) in enumerate(qc.data):
        qubits = [qc.find_bit(q).index for q in qargs]
        print(f"python idx={idx:2d} | paper gate={idx+1:2d} | {instr.name:4s} | qubits={qubits} | params={instr.params}")
    print()

    candidates = build_candidate_semantics()

    print("=== SEARCHING READOUT CONVENTIONS ===")
    results = []
    for sem in candidates:
        print(f"Running: {sem.name}")
        res = evaluate_dataset(
            qc=qc,
            X=X,
            y=y,
            theta_values=THETA_TRAINED,
            semantics=sem,
            n_repeats=50,
        )
        results.append(res)
        print(f"  -> accuracy = {res['mean_acc']:.3f} ± {res['std_acc']:.3f}")

    results_sorted = sorted(results, key=lambda r: abs(r["mean_acc"] - 0.80))

    print("\n=== TOP 10 CLOSEST TO PAPER TARGET (0.80) ===")
    for rank, res in enumerate(results_sorted[:10], start=1):
        print(f"{rank:2d}. {res['name']}")
        print(f"    mean ± std = {res['mean_acc']:.3f} ± {res['std_acc']:.3f}")

    best = results_sorted[0]
    print("\n=== BEST MATCH ===")
    print(best["name"])
    print(f"mean ± std = {best['mean_acc']:.3f} ± {best['std_acc']:.3f}")
    print(f"all accuracies = {best['all_accs']}")


if __name__ == "__main__":
    main()