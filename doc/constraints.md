# Constraints: ChoreoAI

## Technical Constraints

### Programming Stack

Required tools:

```
Python
PyTorch / TensorFlow
NumPy
Pandas
Matplotlib
Git
```

---

### Compute Constraints

Typical training hardware:

```
1 GPU (8–16 GB VRAM)
```

Dataset size must remain manageable.

Target:

```
< 50 GB dataset
```

---

### Pose Estimation Limits

Pose estimation systems suffer from:

* occlusion errors
* depth ambiguity
* joint misidentification

Therefore preprocessing must include:

```
pose smoothing
temporal filtering
joint interpolation
```

---

### Motion Generation Constraints

Unconstrained generation can produce:

```
impossible joint angles
broken skeletons
limb stretching
```

Mitigation techniques:

* joint-angle regularization
* kinematic constraints
* skeleton normalization

---

### Dataset Bias

Dance datasets may bias toward:

```
upright dancing
western dance styles
limited movement diversity
```

Mitigation:

* include diverse dance forms
* include floorwork and nonstandard poses

---

### Multimodal Alignment Difficulty

Some modalities have weak semantic relationships.

Example:

```
architecture → motion
```

Semi-supervised pairing will be required.
