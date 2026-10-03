# NCA All Transitions Data Schema

Complete reference for the `nca_all_transitions.npz` file structure.

## File Format

A `.npz` is a **zip archive of named NumPy arrays**. Load it with:

```python
data = np.load("nca_all_transitions.npz", allow_pickle=True)
```

This returns a dict-like object; access each array by key.

---

## Array Contents

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `states` | `(P, T, C, H, W)` | `float32` | The actual growth/transition frames |
| `from_layout_ids` | `(P,)` | `int64` | Which layout each trajectory started as |
| `to_layout_ids` | `(P,)` | `int64` | Which layout each trajectory ended as |
| `layout_names` | `(N,)` | `<U...>` (string) | Original filenames, indexed by layout id |
| `switch_step` | scalar `()` | `int64` | The single step index at which conditioning switched |
| `cond_vectors` | `(N, COND_DIM)` | `float32` | Learned embedding vector per layout |

---

## Dimension Reference

- **P** = Number of pairs = `NUM_LAYOUTS × (NUM_LAYOUTS − 1)` (all ordered pairs where from ≠ to)  
  *Example: 4 layouts → P = 12*

- **T** = Number of timesteps (default: `150`) — NCA update steps recorded per trajectory

- **C** = Number of channels  
  - `4` if `include_hidden=False` **(RGBA only, default)**
  - `16` if `include_hidden=True` **(full state + hidden channels)**

- **H, W** = Grid size (typically `64 × 64`)

- **N** = Total distinct layouts (not pairs)

- **COND_DIM** = Conditioning embedding size (e.g., `8` in typical configs)

---

## Indexing and Data Access

### Accessing a Trajectory

`states[i]` is **one full trajectory**. To retrieve complete information about it:

```python
data = np.load("nca_all_transitions.npz", allow_pickle=True)

i = 5  # Example trajectory index
trajectory = data["states"][i]              # shape (150, 4, 64, 64)
from_id = data["from_layout_ids"][i]        # e.g. 1
to_id = data["to_layout_ids"][i]            # e.g. 3
from_name = data["layout_names"][from_id]   # e.g. "room_a.png"
to_name = data["layout_names"][to_id]       # e.g. "room_c.png"
```

### Index Structure

**`from_layout_ids` and `to_layout_ids` are parallel arrays:**
- Index `i` into both to get the layout pair that produced `states[i]`
- This is a "structure of arrays" layout (flat, fast-to-load NumPy arrays)
- More efficient than a dict per entry

**`layout_names` and `cond_vectors` use a different index:**
- Indexed by raw layout id (`0` to `N−1`), not by pair index
- `layout_names[from_id]` and `cond_vectors[to_id]` are lookups
- Separate indexing from the `states`/`from_layout_ids`/`to_layout_ids` triple

---

## Frame Structure

### Single Frame Access

`states[i, t]` is **one frame** with shape `(4, 64, 64)` if RGBA-only.

### Channel Breakdown

| Channel | Type | Details |
|---------|------|---------|
| **0–2** | R, G, B | Premultiplied by alpha (matches how targets were loaded during training) |
| **3** | Alpha | The "alive" mask — controls visibility/activation |

> **Note:** For `include_hidden=True` configurations, channels 4–15 contain the full NCA hidden state beyond RGBA.