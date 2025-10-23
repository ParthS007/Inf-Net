# Inf-Net Training System

## Training Scripts

### 1. `MyTrain_LungInf_Morph.py`
**Purpose**: Standard Inf-Net and Inf-Net with optional morphology

**Features**:
- Standard Inf-Net training
- Optional morphological operations (open, close, dilation, erosion)
- Batch size configuration (32, 64, 128)
- Multi-run support for reproducibility

**Arguments**:
```bash
--batchsize 32              # Batch size
--run 1                     # Run number (1-3)
--enable_morphology         # Enable morphology
--morph_operation close     # Morphology operation
--morph_kernel_size 3       # Kernel size for morphology
```

**Snapshot Path Structure**:
```
Snapshots/save_weights/
├── Inf-Net/batch_32/run_1/
├── Inf-Net/batch_64/run_1/
└── Inf-Net_Morph/close/batch_32/run_1/
```

---

### 2. `MyTrain_LungInfDP_Morph.py`
**Purpose**: Inf-Net with differential privacy and optional morphology

**Features**:
- Differential privacy with configurable epsilon
- Optional morphological operations with DP
- Opacus-based privacy accounting
- Noise multiplier to epsilon mapping

**Arguments**:
```bash
--batchsize 32                    # Batch size
--run 1                           # Run number
--enable_privacy                  # Enable DP training
--noise_multiplier 1.5            # Noise multiplier for DP-SGD
--delta 1e-5                      # Privacy parameter (δ)
--enable_morphology               # Enable morphology with DP
--morph_operation open            # Morphology operation
--morph_privacy_budget 1.0        # Privacy budget for morphology
```

**Snapshot Path Structure**:
```
Snapshots/save_weights/
├── Inf-Net_DP/batch_32/run_1/
├── Inf-Net_DP/batch_64/run_1/
└── Inf-Net_DP_Morph/open/batch_32/run_1/
```

---

## Job Submission System

### 1. `submit_training_jobs.py`
Python-based job generator that creates SBATCH scripts for all configurations.

**Generates Configurations**:
- **Standard**: 3 batch sizes × 3 runs = 9 jobs
- **Morph**: 3 batch sizes × 4 operations × 3 runs = 36 jobs
- **DP**: 3 batch sizes × 3 epsilon values × 3 runs = 27 jobs
- **DP+Morph**: 3 batch sizes × 3 epsilon × 4 operations × 3 runs = 108 jobs
- **Total**: 180 jobs for comprehensive study

---

## Snapshot Directory Structure

### Complete Hierarchy
```
Snapshots/save_weights/
│
├── Inf-Net/                          # Standard Inf-Net
│   ├── batch_32/
│   │   ├── run_1/
│   │   │   ├── Inf-Net-10.pth
│   │   │   ├── Inf-Net-20.pth
│   │   │   └── ...
│   │   ├── run_2/
│   │   └── run_3/
│   ├── batch_64/
│   │   ├── run_1/
│   │   ├── run_2/
│   │   └── run_3/
│   └── batch_128/
│       ├── run_1/
│       ├── run_2/
│       └── run_3/
│
├── Inf-Net_Morph/                    # Inf-Net with Morphology
│   ├── open/
│   │   ├── batch_32/
│   │   │   ├── run_1/
│   │   │   ├── run_2/
│   │   │   └── run_3/
│   │   ├── batch_64/
│   │   └── batch_128/
│   ├── close/
│   ├── dilation/
│   └── erosion/
│
├── Inf-Net_DP/                       # Inf-Net with Differential Privacy
│   ├── batch_32/
│   │   ├── run_1/
│   │   ├── run_2/
│   │   └── run_3/
│   ├── batch_64/
│   └── batch_128/
│
└── Inf-Net_DP_Morph/                 # Inf-Net with DP + Morphology
    ├── open/
    │   ├── batch_32/
    │   │   ├── run_1/
    │   │   ├── run_2/
    │   │   └── run_3/
    │   ├── batch_64/
    │   └── batch_128/
    ├── close/
    ├── dilation/
    └── erosion/
```
---

## Configuration Mapping

### Batch Sizes
- 32: More frequent updates, slower convergence
- 64: Balanced
- 128: Fewer updates, faster epochs

### Morphology Operations
- `open`: Erosion → Dilation (removes small noise)
- `close`: Dilation → Erosion (fills holes)
- `dilation`: Expands white regions
- `erosion`: Shrinks white regions

### Epsilon Values (Differential Privacy)
| ε Value | Privacy | Accuracy | Use Case |
|---------|---------|----------|----------|
| 1 | Strong | Lower | Research requiring high privacy |
| 8 | Medium | Medium | Balanced privacy-accuracy |
| 200 | Weak | Higher | Non-critical applications |
