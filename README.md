# Federated Learning DES Simulator

Simulatore **Discrete Event** per Federated Learning che permette di:

- simulare il tempo di **training** e **comunicazione** per ogni client,
- scegliere se **addestrare un dummy model** oppure simulare solo i tempi,
- usare **distribuzioni arbitrarie** o un **modello di ML** per predire i tempi di training/communication,
- configurare **hardware, GPU e banda per device** via file TOML.

---

## Passi principali

1. (Opzionale) **Genera il dataset** per il modello ML dei tempi (PyTorch):

   ```bash
   python scripts/generate_training_time_dataset_torch.py \
     --output data/training_time_dataset.csv \
     --n-samples 2000
   # ripeti con --append e seed diversi per arrivare a ≥ 10000 righe
   ```

2. (Opzionale) **Addestra il modello ML** di tempo:

   ```bash
   python -m scripts.train_ml_time_model
   ```

   Questo crea `data/ml_time_model.npz`, usato quando `training_time.kind = "ml"`.

3. **Scegli una config** in `configs/`:

   - `example1_no_model_dist.toml` – niente modello, tempi da distribuzioni.
   - `example2_no_model_ml.toml`   – niente modello, tempi da modello ML.
   - `example3_dummy_dist.toml`    – dummy model addestrato, tempi da distribuzioni.
   - `example4_dummy_ml.toml`      – dummy model addestrato, tempi da modello ML.

4. **Lancia la simulazione**, ad esempio:

   ```bash
   python -m fl_sim.cli --config configs/example4_dummy_ml.toml
   ```

   I risultati vanno in `output_*`:

   - `client_times.csv`  – tempi per round/client (training, communication, total)
   - `summary.json`      – statistiche per client e metriche del dummy model
   - `histograms.json`   – istogrammi per client

5. **Plotta i risultati**:

   ```bash
   python scripts/plot_results.py --dir output_dummy_ml
   ```

   Questo genera:

   - `plot_hist_training_comm_all.png`
   - `plot_avg_time_per_client.png`
   - `plot_model_metrics.png` (se il dummy model è abilitato)

Esempi di `plot_avg_time_per_client.png` (800×400), con training (blu) + communication (rosso) per client:

![Average training + communication time per client (molti client)](output_dummy_ml/plot_avg_time_per_client.png)

![Average training + communication time per client (pochi client)](output_no_model_ml/plot_avg_time_per_client.png)

Le barre sono ordinate per **tempo totale medio** crescente; puoi usare queste viste con:

- dummy model addestrato,
- sole distribuzioni,
- modello ML per predire il tempo di training e di comunicazione per ogni client.
