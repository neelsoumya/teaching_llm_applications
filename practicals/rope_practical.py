"""
PRACTICAL: Rotary Position Embedding (RoPE)
============================================

Goal of this practical
-----------------------
In the lecture slide "Implementation and code for RoPE" we saw four steps
inside a real transformer's attention block:

    query_states = self.q_proj(hidden_states)
    key_states   = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states   = key_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)          # <- "Get the RoPE matrix cos/sin"
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)  # <- "Multiply query/key inputs"

    ... (same stuff as usual multi-head self attention below)

This file reproduces ONLY the two RoPE-specific lines (the highlighted ones)
in plain, from-scratch NumPy, with no deep learning framework, so that
students can see exactly what "rotary_emb" and "apply_rotary_pos_emb" do
under the hood.

We deliberately DO NOT build a full transformer here. We build the smallest
possible example that still shows the real behaviour: a single query vector,
a single key vector, and a handful of positions.

Run this file top to bottom: `python rope_practical.py`
Each PART prints output so you can follow along in class.
"""

import numpy as np

np.set_printoptions(precision=3, suppress=True)


# ---------------------------------------------------------------------------
# PART 0: Why do we need this at all?
# ---------------------------------------------------------------------------
# Self-attention computes a dot product between a query vector q (from the
# current token) and a key vector k (from every other token) to decide how
# much attention to pay. But q and k, as produced by q_proj/k_proj, carry NO
# information about WHERE the token is in the sequence. Attention would treat
# "the cat sat" exactly the same as "sat the cat" - order would not matter.
#
# RoPE fixes this by ROTATING the query and key vectors by an angle that
# depends on the token's position, BEFORE the dot product is taken. Rotating
# is a good choice because a rotation does not change the *length* of a
# vector, so it does not blow up the training dynamics.
#
# The slide's note says it best:
#     "embedding at EACH attention operation to enforce position invariance"
# RoPE is re-applied every single time attention is computed (every layer,
# every head) - it is not a one-off input embedding like the original
# Transformer's positional encodings.


# ---------------------------------------------------------------------------
# PART 1: Build the RoPE frequencies (this is what self.rotary_emb does)
# ---------------------------------------------------------------------------
def get_rope_frequencies(head_dim: int, base: float = 10000.0) -> np.ndarray:
    """
    RoPE splits a head_dim-sized vector into head_dim/2 PAIRS of numbers.
    Each pair is treated as a 2D point (x, y) and gets rotated by its own
    angle. Pairs that come later in the vector rotate more SLOWLY.

    This function returns one base frequency per pair.
    Shape: (head_dim / 2,)
    """
    assert head_dim % 2 == 0, "head_dim must be even, since we rotate PAIRS"
    i = np.arange(0, head_dim, 2)               # 0, 2, 4, ..., head_dim-2
    freqs = 1.0 / (base ** (i / head_dim))       # one frequency per pair
    return freqs


def rotary_emb(head_dim: int, position_ids: np.ndarray, base: float = 10000.0):
    """
    This is the from-scratch equivalent of `self.rotary_emb(value_states, position_ids)`
    from the slide. Real code passes `value_states` only to read off the dtype
    and device - the actual math only needs `position_ids` and `head_dim`.

    For every position m in position_ids, and every pair-frequency f,
    the rotation angle is simply: angle = m * f

    Returns cos and sin, each of shape (num_positions, head_dim), where the
    angle for each pair has been duplicated twice so it lines up with both
    the x and the y coordinate of that pair (see PART 2).
    """
    freqs = get_rope_frequencies(head_dim, base)         # (head_dim/2,)
    position_ids = position_ids.astype(np.float64)       # (num_positions,)

    # Outer product: angle[m, j] = position_ids[m] * freqs[j]
    angles = np.outer(position_ids, freqs)                # (num_positions, head_dim/2)

    # Duplicate each angle so the cos/sin table has one entry per DIMENSION
    # (not per pair). This matches how HuggingFace-style code concatenates
    # [angles, angles] along the last axis.
    angles = np.concatenate([angles, angles], axis=-1)     # (num_positions, head_dim)

    cos = np.cos(angles)
    sin = np.sin(angles)
    return cos, sin


# ---------------------------------------------------------------------------
# PART 2: Apply the rotation to q and k (this is apply_rotary_pos_emb)
# ---------------------------------------------------------------------------
def rotate_half(x: np.ndarray) -> np.ndarray:
    """
    Helper used inside apply_rotary_pos_emb.

    Splits the last dimension of x into two halves [x1, x2] and returns
    [-x2, x1]. This "swap and negate" trick, combined with the duplicated
    cos/sin table from PART 1, is a fast way to rotate every (x1_j, x2_j)
    pair by its own angle WITHOUT writing a slow Python for-loop over pairs.

    Why it's equivalent to a 2D rotation:
        A standard 2D rotation of point (x, y) by angle theta is:
            x' = x*cos(theta) - y*sin(theta)
            y' = x*sin(theta) + y*cos(theta)
        If we let x = x1_j and y = x2_j, then:
            x' = x1*cos - x2*sin
            y' = x2*cos + x1*sin
        which is exactly  x * cos + rotate_half(x) * sin  (see below).
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q: np.ndarray, k: np.ndarray, cos: np.ndarray, sin: np.ndarray):
    """
    Direct equivalent of:
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    q, k   : shape (..., head_dim)  -- one vector per token
    cos,sin: shape (num_positions, head_dim) -- one row per position, from PART 1

    This is THE formula. Everything above was just building its ingredients.
    """
    q_rotated = (q * cos) + (rotate_half(q) * sin)
    k_rotated = (k * cos) + (rotate_half(k) * sin)
    return q_rotated, k_rotated


# ---------------------------------------------------------------------------
# PART 3: A tiny worked example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    head_dim = 4  # kept tiny on purpose so every number is easy to see
    num_positions = 3
    position_ids = np.arange(num_positions)   # positions 0, 1, 2

    print("=" * 70)
    print("PART 3a: cos/sin table for head_dim=4, positions=[0,1,2]")
    print("=" * 70)
    cos, sin = rotary_emb(head_dim, position_ids)
    print("cos:\n", cos)
    print("sin:\n", sin)
    print(
        "\nNotice row 0 (position 0) is always cos=1, sin=0: rotating by\n"
        "angle 0 does nothing. That's why the token at position 0 is\n"
        "unaffected by RoPE, while later positions are rotated more."
    )

    print("\n" + "=" * 70)
    print("PART 3b: same query vector, three different positions")
    print("=" * 70)
    # Pretend q_proj produced the SAME query vector for every position.
    # In real attention this would be unusual, but it isolates exactly what
    # RoPE contributes: with no RoPE, these three rows would be identical.
    same_q = np.tile(np.array([1.0, 0.5, -0.3, 0.8]), (num_positions, 1))
    same_k = same_q.copy()

    rotated_q, rotated_k = apply_rotary_pos_emb(same_q, same_k, cos, sin)
    print("Original q (identical at every position):\n", same_q)
    print("\nRotated q (now position-dependent!):\n", rotated_q)

    print("\n" + "=" * 70)
    print("PART 3c: the key property RoPE is famous for")
    print("=" * 70)
    print(
        "The dot product between a rotated query at position m and a\n"
        "rotated key at position n depends ONLY on the distance (m - n),\n"
        "never on the absolute positions m and n themselves. Let's check:"
    )

    def rope_dot(q_vec, k_vec, m, n, head_dim=4, base=10000.0):
        cos_mn, sin_mn = rotary_emb(head_dim, np.array([m, n]), base)
        q_rot, k_rot = apply_rotary_pos_emb(
            np.array([q_vec, q_vec]), np.array([k_vec, k_vec]), cos_mn, sin_mn
        )
        q_at_m = q_rot[0]
        k_at_n = k_rot[1]
        return float(np.dot(q_at_m, k_at_n))

    q_vec = np.array([1.0, 0.5, -0.3, 0.8])
    k_vec = np.array([0.2, -0.7, 0.4, 0.1])

    # Same relative distance (m - n = 2), different absolute positions
    dot_a = rope_dot(q_vec, k_vec, m=5, n=3)     # distance 2
    dot_b = rope_dot(q_vec, k_vec, m=100, n=98)  # distance 2, far away
    dot_c = rope_dot(q_vec, k_vec, m=1, n=0)     # distance 1, different!

    print(f"dot(q@pos=5,   k@pos=3)   [distance=2]  -> {dot_a:.6f}")
    print(f"dot(q@pos=100, k@pos=98)  [distance=2]  -> {dot_b:.6f}")
    print(f"dot(q@pos=1,   k@pos=0)   [distance=1]  -> {dot_c:.6f}")
    print(
        "\n=> The first two numbers match (same distance, far apart in\n"
        "absolute terms) and the third differs (different distance).\n"
        "This is 'relative position encoding' emerging naturally from a\n"
        "rotation, with no extra parameters and no lookup table."
    )


# ---------------------------------------------------------------------------
# EXERCISES FOR STUDENTS (uncomment and fill in the TODOs)
# ---------------------------------------------------------------------------
# Exercise 1
# ----------
# The slide highlights `position_ids` as the extra input RoPE needs beyond
# ordinary attention. Modify PART 3b so that instead of positions [0, 1, 2],
# you use positions [0, 10, 20]. Do the rotated vectors change more or less
# between steps compared to [0, 1, 2]? Why?
#
# Exercise 2
# ----------
# `base=10000.0` controls how quickly the rotation angles grow across
# dimensions. Re-run PART 3c's distance check with base=100.0 instead of
# 10000.0. Do dot_a and dot_b still match each other? Do the absolute values
# of the dot products change? What does this tell you about what `base`
# controls?
#
# Exercise 3 (harder)
# --------------------
# head_dim must be even because RoPE rotates PAIRS of dimensions. Try setting
# head_dim=5 and see what error you get. Then write one sentence in a comment
# explaining, in your own words, why an odd head_dim is not allowed.
#
# Exercise 4 (connects back to the slide)
# ----------------------------------------
# On the slide, `apply_rotary_pos_emb` is called on BOTH query_states and
# key_states using the SAME cos/sin. In PART 3c's `rope_dot` helper, try
# instead rotating q with position m but rotating k with a DIFFERENT base
# (e.g. base=50.0 for q, base=10000.0 for k). Does the "distance-only"
# property from PART 3c still hold? What does this tell you about why real
# implementations always use one shared rotary_emb call for the whole layer?
