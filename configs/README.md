# Simulator configs (TOML)

This folder contains the TOML files that configure simulations for the Federated Learning DES simulator.

Each config is made of the following main sections:

- `[simulation]` – global simulation parameters.
- `[training_time]` – time model for **training**.
- `[communication_time]` – time model for **communication**.
- `[output]` – where results are written.
- `[model]` – (optional) dummy FL model and time‑measurement options.
- `[devices."<id>"]` – (optional) per‑device overrides for hardware/bandwidth/batch size.

Below is a detailed description of each section.

---

## 1. `[simulation]` section

Example:

```toml
[simulation]
rounds = 100          # number of federated rounds
num_clients = 50      # total number of clients
clients_per_round = 10  # how many clients per round
seed = 1234           # RNG seed
```

Fields:

- `rounds` (int): number of federated rounds.
- `num_clients` (int): total number of clients (IDs from `0` to `num_clients-1`).
- `clients_per_round` (int, optional): number of clients per round.
  - If omitted, `participation_rate` is used instead (default 1.0).
- `participation_rate` (float, optional): fraction of clients selected each round.
- `seed` (int): global RNG seed.

---

## 2. `[training_time]` section

Controls how **training times** are generated for each client.

Main field:

- `kind` (string): time model type. Supported values:
  - `"lognormal"` / `"normal"` / `"exponential"` / `"constant"` – simple parametric models.
  - `"trace"` – replay times from a CSV trace file.
  - `"hardware"` – analytic model based on hardware + model metadata.
  - `"ml"` – ML time model trained from `data/ml_time_model.npz`.

Examples:

```toml
# Example 1: pure lognormal
[training_time]
kind = "lognormal"
mean = 0.0
sigma = 0.5

# Example 2: ML time model
[training_time]
kind = "ml"
mean = 0.0      # extra lognormal noise (mu in log-space)
sigma = 0.3
```

Common fields:

- `mean`, `sigma` (float): base parameters for lognormal/normal noise.
- `constant` (float): for `kind = "constant"`, fixed time.
- `file` / `column` (string): for `kind = "trace"`, CSV path/column.

Notes for `kind = "ml"`:

- requires `data/ml_time_model.npz` (from `scripts/train_ml_time_model.py`).
- uses as features:
  - hardware: `cpu_cores`, `cpu_freq_ghz`, `mem_available_gb`, `accelerator_tflops`,
  - model: `flops_per_sample`, `model_size_bytes`, `samples_per_client`,
  - training: `batch_size`, `local_epochs`.
- these values are taken from `[model]` and, partly, from `[devices."<id>"]`.

---

## 3. `[communication_time]` section

Controls **communication times** (upload of gradients/model).

Main field:

- `kind` (string): time model. Most useful cases:
  - `"bandwidth"` – time ≈ gradient_size / bandwidth (stochastic).
  - `"lognormal"`, `"normal"`, `"constant"`, etc. – purely parametric models.

Typical example:

```toml
[communication_time]
kind = "bandwidth"
mean = 0.0
sigma = 0.3
default_bandwidth_mean_mbps = 20.0   # banda media di default per tutti i device (Mbps)
default_bandwidth_jitter = 0.3       # jitter lognormale di default
```

For `kind = "bandwidth"`:

- for each client:
  - `payload_bytes ≈ model_size_bytes` (from `[model]`);
  - `bandwidth ~ LogNormal(mu, sigma)` in Mbps;
  - `time ≈ payload_bytes / bandwidth_bps`.
- Bandwidth is determined as:
  1. If `[devices."<id>"].bandwidth_mean_mbps` / `bandwidth_jitter` exist, use them.
  2. Otherwise use `default_bandwidth_mean_mbps` / `default_bandwidth_jitter` when present.
  3. If nothing is specified, internal defaults are used: 10 Mbps, jitter 0.4.

---

## 4. `[output]` section

Specifies where results are written.

```toml
[output]
directory = "output_dummy_ml"
write_traces_csv = true
write_summary_json = true
write_histograms_json = true
per_client_histograms = true
histogram_bins = 20
```

Fields:

- `directory` (string): output folder.
- `write_traces_csv` (bool): if `true`, writes `client_times.csv`.
- `write_summary_json` (bool): if `true`, writes `summary.json`.
- `write_histograms_json` (bool): if `true`, writes `histograms.json`.
- `per_client_histograms` (bool): enable per‑client histograms.
- `histogram_bins` (int): number of bins.

---

## 5. `[model]` section (dummy FL model)

Controls whether and how a **dummy model** (logreg‑like) is trained during simulation.

Example:

```toml
[model]
enabled = true              # if false: no dummy training
measure_training_time = false  # if true: use wall-clock training time as training_time
kind = "dummy_logreg"

input_dim = 10
num_classes = 3
samples_per_client = 200
test_samples = 500
local_epochs = 1
batch_size = 128
learning_rate = 0.1
eval_every_rounds = 5
flops_per_sample = 1000000.0
model_size_bytes = 5000000
```

Key fields:

- `enabled` (bool):
  - `true`  → dummy model is instantiated and trained; accuracy/macro‑F1 are reported.
  - `false` → **no** model is trained (values below are still used by time models).
- `measure_training_time` (bool):
  - `false` (default): training times come from the time model (`[training_time]`).
    - dummy training (if `enabled`) is used only for metrics.
  - `true`: simulator measures the **wall‑clock** time of `local_train()` per client and uses it as `training_time` (overriding the time model).
- `flops_per_sample`, `model_size_bytes`, `samples_per_client`, `batch_size`, `local_epochs`:
  - describe how “heavy” the model and training are;
  - used by:
    - the ML time model (`kind = "ml"`),
    - the analytic model (`kind = "hardware"`),
    - communication model (gradient payload size).

If `[model]` is missing:

- `cfg.model = None` and no dummy model is built.

---

## 6. `[devices."<id>"]` sections (per‑device)

Customize hardware, GPU, bandwidth and batch size for individual clients.

Example:

```toml
[devices."0"]
cpu_cores = 4
cpu_freq_ghz = 2.5
mem_total_gb = 8
mem_available_gb = 4
accelerator_tflops = 0.0
has_gpu = false
bandwidth_mean_mbps = 5.0
bandwidth_jitter = 0.5
batch_size = 64         # local batch size used in ML time model features

[devices."1"]
cpu_cores = 16
has_gpu = true
accelerator_tflops = 10.0
bandwidth_mean_mbps = 50.0
bandwidth_jitter = 0.2
batch_size = 256
```

Fields:

- `cpu_cores`, `cpu_freq_ghz`, `mem_total_gb`, `mem_available_gb`:
  - override host hardware.
- `accelerator_tflops`:
  - GPU/accelerator TFLOPs for this device.
- `has_gpu`:
  - `true` / `false`: explicitly mark device as having a GPU.
  - if omitted, inferred from `accelerator_tflops > 0` or host.
- `bandwidth_mean_mbps`, `bandwidth_jitter`:
  - mean bandwidth and lognormal jitter for `"bandwidth"` model.
- `batch_size`:
  - per‑device batch size;
  - used in the ML time model features (`log_batch_size`).

Clients **not** listed in `[devices."<id>"]` use:

- host hardware (`collect_host_hardware()`),
- default bandwidth (`default_bandwidth_mean_mbps` / `default_bandwidth_jitter` in `[communication_time]`),
- global `batch_size` from `[model]`.

---

## 7. Ready‑made configs

This folder contains ready‑to‑use examples:

- `example1_no_model_dist.toml`  
  No model, times from distributions (training + communication).

- `example2_no_model_ml.toml`  
  No model, training times from ML model, communication from `"bandwidth"`.

- `example3_dummy_dist.toml`  
  Dummy model trained, training times from distributions, communication `"bandwidth"`.

- `example4_dummy_ml.toml`  
  Dummy model trained, training times from ML model, communication `"bandwidth"`.

You can use them as templates for your own configs.  
To run a simulation:

```bash
python -m fl_sim.cli --config configs/example4_dummy_ml.toml
```
