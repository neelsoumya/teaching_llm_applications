"""
INSTRUCTOR SOLUTIONS: rope_practical.py exercises
==================================================
Companion file - not meant to be handed to students before the exercise.
Each solution is self-contained and re-imports from rope_practical.py.
"""

import numpy as np
from rope_practical import get_rope_frequencies, rotary_emb, apply_rotary_pos_emb, rotate_half

np.set_printoptions(precision=3, suppress=True)


def exercise_1():
    """Positions [0, 10, 20] instead of [0, 1, 2]."""
    print("EXERCISE 1: wider position gaps")
    print("-" * 50)
    head_dim = 4
    same_q = np.tile(np.array([1.0, 0.5, -0.3, 0.8]), (3, 1))

    for positions in [np.array([0, 1, 2]), np.array([0, 10, 20])]:
        cos, sin = rotary_emb(head_dim, positions)
        rotated_q, _ = apply_rotary_pos_emb(same_q, same_q, cos, sin)
        print(f"positions={positions.tolist()}")
        print(rotated_q, "\n")

    print(
        "ANSWER: with positions [0, 10, 20] the vectors change MUCH more\n"
        "between rows, because the rotation angle is position * frequency -\n"
        "bigger position gaps mean bigger angle gaps, so the rotated\n"
        "vectors are less similar to each other (more 'distinguishable').\n"
    )


def exercise_2():
    """base=100.0 instead of base=10000.0."""
    print("EXERCISE 2: changing base")
    print("-" * 50)
    head_dim = 4
    q_vec = np.array([1.0, 0.5, -0.3, 0.8])
    k_vec = np.array([0.2, -0.7, 0.4, 0.1])

    def rope_dot(m, n, base):
        cos_mn, sin_mn = rotary_emb(head_dim, np.array([m, n]), base=base)
        q_rot, k_rot = apply_rotary_pos_emb(
            np.array([q_vec, q_vec]), np.array([k_vec, k_vec]), cos_mn, sin_mn
        )
        return float(np.dot(q_rot[0], k_rot[1]))

    for base in [10000.0, 100.0]:
        dot_a = rope_dot(5, 3, base)
        dot_b = rope_dot(100, 98, base)
        print(f"base={base}: dot(pos5,pos3)={dot_a:.6f}  dot(pos100,pos98)={dot_b:.6f}")

    print(
        "\nANSWER: dot_a and dot_b still match each other for ANY base - the\n"
        "distance-only property does not depend on base. What changes is\n"
        "the ACTUAL VALUE of the dot product: a smaller base makes angles\n"
        "grow faster across dimensions and across positions, so rotations\n"
        "'wrap around' sooner. 'base' controls the range of positions over\n"
        "which RoPE can distinguish tokens before angles start repeating -\n"
        "this is directly related to how well a model generalises to\n"
        "longer sequences than it was trained on.\n"
    )


def exercise_3():
    """head_dim=5 (odd) should fail."""
    print("EXERCISE 3: odd head_dim")
    print("-" * 50)
    try:
        get_rope_frequencies(5)
    except AssertionError as e:
        print(f"Got expected error: {e}")
    print(
        "\nANSWER: RoPE rotates 2D PAIRS of dimensions at a time (each pair\n"
        "is treated as an (x, y) point on a 2D plane and rotated by an\n"
        "angle). An odd number of dimensions means one dimension would be\n"
        "left over with no partner to pair with, so it could not be rotated.\n"
    )


def exercise_4():
    """Mismatched base for q vs k."""
    print("EXERCISE 4: mismatched rotary_emb calls for q and k")
    print("-" * 50)
    head_dim = 4
    q_vec = np.array([1.0, 0.5, -0.3, 0.8])
    k_vec = np.array([0.2, -0.7, 0.4, 0.1])

    def mismatched_rope_dot(m, n, base_q, base_k):
        cos_q, sin_q = rotary_emb(head_dim, np.array([m]), base=base_q)
        cos_k, sin_k = rotary_emb(head_dim, np.array([n]), base=base_k)
        q_rot = (q_vec * cos_q[0]) + (rotate_half(q_vec) * sin_q[0])
        k_rot = (k_vec * cos_k[0]) + (rotate_half(k_vec) * sin_k[0])
        return float(np.dot(q_rot, k_rot))

    dot_a = mismatched_rope_dot(5, 3, base_q=50.0, base_k=10000.0)
    dot_b = mismatched_rope_dot(100, 98, base_q=50.0, base_k=10000.0)
    print(f"mismatched bases: dot(pos5,pos3)={dot_a:.6f}  dot(pos100,pos98)={dot_b:.6f}")

    print(
        "\nANSWER: the distance-only property BREAKS - the two dot products\n"
        "no longer match, even though both pairs have the same distance=2.\n"
        "This is why real code calls self.rotary_emb ONCE per layer and\n"
        "passes the SAME cos/sin into apply_rotary_pos_emb for both query\n"
        "and key: the elegant relative-position math only works when q and\n"
        "k are rotated using the exact same angle convention.\n"
    )


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
