# Federated Learning DES Simulator

Discrete‑event simulator for Federated Learning. It lets you:

- simulate **training** and **communication** time per client,
- optionally **train a dummy model** (for accuracy/F1) or simulate only times,
- use **arbitrary distributions** or an **ML model** to predict training/communication time,
- configure **hardware, GPU and bandwidth per device** via TOML configs.

---

## Quick workflow

1. (Optional) **Generate a dataset** for the ML time model (PyTorch):

   ```bash
   python scripts/generate_training_time_dataset_torch.py \
     --output data/training_time_dataset.csv \
     --n-samples 2000
   # repeat with --append and different --seed to reach ≥ 10000 rows
   ```

2. (Optional) **Train the ML time model**:

   ```bash
   python -m scripts.train_ml_time_model
   ```

   This creates `data/ml_time_model.npz`, used when `training_time.kind = "ml"`.

3. **Pick a config** from `configs/`:

   - `example1_no_model_dist.toml` – no model, times from distributions.
   - `example2_no_model_ml.toml`   – no model, times from ML model.
   - `example3_dummy_dist.toml`    – dummy model trained, times from distributions.
   - `example4_dummy_ml.toml`      – dummy model trained, times from ML model.

4. **Run a simulation**, e.g.:

   ```bash
   python -m fl_sim.cli --config configs/example4_dummy_ml.toml
   ```

   Outputs go to `output_*`:

   - `client_times.csv`  – per‑round per‑client times (training, communication, total)
   - `summary.json`      – per‑client stats and dummy model metrics
   - `histograms.json`   – per‑client histograms

5. **Plot results**:

   ```bash
   python scripts/plot_results.py --dir output_dummy_ml
   ```

   This generates:

   - `plot_hist_training_comm_all.png`
   - `plot_avg_time_per_client.png`
   - `plot_model_metrics.png` (when the dummy model is enabled)

Examples of `plot_avg_time_per_client.png` (800×400), showing blue training time and red communication time per client:

![Average training + communication time per client (many clients)](output_dummy_ml/plot_avg_time_per_client.png)

![Average training + communication time per client (few clients)](output_no_model_ml/plot_avg_time_per_client.png)

Bars are sorted by **total mean time**; you can use these plots for scenarios where:

- a dummy model is trained,
- only distributions are used,
- an ML model predicts both training and communication time for each client.
