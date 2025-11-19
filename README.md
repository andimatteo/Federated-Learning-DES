# Federated Learning DES Simulator

Questo progetto implementa un simulatore **Discrete Event** per Federated Learning, con:

- modellazione del tempo di **training** per client (anche via modello ML),
- modellazione del tempo di **comunicazione** basata sulla **banda** per dispositivo,
- metriche di accuracy / macro-F1 di un modello dummy (logreg-like),
- generazione ed utilizzo di un **dataset reale** di tempi di training.

Di seguito i comandi principali per:

1. generare / espandere il dataset per il modello ML,
2. addestrare il modello di tempo ML,
3. lanciare la simulazione con `training_time.kind = "ml"` e modello di banda per la comunicazione,
4. plottare i risultati.

## 1. Generare il dataset di tempi di training (PyTorch)

Script autocontenuto (può essere copiato su altre macchine):

- `scripts/generate_training_time_dataset_torch.py`

Esempio: genera 2000 configurazioni e salva in `data/training_time_dataset.csv`:

```bash
python scripts/generate_training_time_dataset_torch.py \
  --output data/training_time_dataset.csv \
  --n-samples 2000
```

Per espandere il dataset (es. fino a ≥ 10000 righe), ripeti più volte con `--append` e seed diversi:

```bash
python scripts/generate_training_time_dataset_torch.py \
  --output data/training_time_dataset.csv \
  --n-samples 2000 --seed 2 --append

python scripts/generate_training_time_dataset_torch.py \
  --output data/training_time_dataset.csv \
  --n-samples 2000 --seed 3 --append
# ... e così via, variando --seed
```

Il CSV risultante in `data/training_time_dataset.csv` contiene, tra le altre:

- `cpu_cores`, `cpu_freq_ghz`, `mem_available_gb`, `accelerator_tflops`
- `flops_per_sample`, `model_size_bytes`
- `samples_per_client`, `batch_size`, `local_epochs`
- `model_family` (logreg / cnn_like / transformer_like), `param_count`
- `training_time_seconds` (tempo stimato, basato su misure reali per step).

## 2. Addestrare il modello ML di tempo di training

Lo script di training legge il dataset da `data/training_time_dataset.csv` e salva i parametri in `data/ml_time_model.npz`:

```bash
python -m scripts.train_ml_time_model
```

Internamente:

- costruisce un vettore di feature per ogni riga, che include:
  - hardware: `cpu_cores`, `cpu_freq_ghz`, `mem_available_gb`, `accelerator_tflops`
  - modello: `log10(flops_per_sample)`, `log10(model_size_bytes)`, `log10(samples_per_client)`
  - training: `log10(batch_size)`, `log10(local_epochs)`
- normalizza le feature,
- allena un piccolo **MLP** (1 hidden layer) per approssimare `training_time_seconds`,
- salva pesi e statistiche in `data/ml_time_model.npz`.

Questo file viene poi caricato dal simulatore quando `training_time.kind = "ml"`.

## 3. Eseguire la simulazione con modello ML per il computing e banda per la comunicazione

Configurazione di esempio: `configs/example_ml.toml`.

Punti chiave della config:

```toml
[simulation]
rounds = 100
num_clients = 50
clients_per_round = 10
seed = 1234

[training_time]
kind = "ml"        # usa il modello ML addestrato in data/ml_time_model.npz
mean = 0.0
sigma = 0.3

[communication_time]
kind = "bandwidth" # tempo di invio = dimensione_gradiente / banda_stocastica
mean = 0.0
sigma = 0.3
default_bandwidth_mean_mbps = 20.0   # banda media di default per tutti i device
default_bandwidth_jitter = 0.3       # jitter di default

[output]
directory = "output_ml"
write_traces_csv = true
write_summary_json = true
write_histograms_json = true
per_client_histograms = true
histogram_bins = 20

[model]
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

[devices."0"]
cpu_cores = 4
bandwidth_mean_mbps = 5.0
bandwidth_jitter = 0.5
has_gpu = false          # opzionale, forza assenza GPU
accelerator_tflops = 0.0 # opzionale, TFLOPs dell'eventuale GPU

[devices."1"]
cpu_cores = 16
bandwidth_mean_mbps = 50.0
bandwidth_jitter = 0.2
has_gpu = true           # opzionale, forza presenza GPU
accelerator_tflops = 10.0

[devices."2"]
cpu_cores = 32
bandwidth_mean_mbps = 1.0
bandwidth_jitter = 0.5
```

- `training_time.kind = "ml"`:
  - il simulatore usa il modello ML per stimare il tempo di training per client, in funzione di hardware e parametri del training.
- `communication_time.kind = "bandwidth"`:
  - il tempo di invio è modellato come:
    - `time ≈ payload_bytes / banda`
    - `banda` è lognormale per device, con media `bandwidth_mean_mbps` e jitter `bandwidth_jitter`.
- `[devices."id"]`:
  - permette di specificare per ogni device:
    - hardware (es. `cpu_cores`, `cpu_freq_ghz`, `mem_*`, `accelerator_tflops`),
    - banda media (`bandwidth_mean_mbps`) e volatilità (`bandwidth_jitter`).
  - i device non elencati usano i valori di default derivati dall’host.

Per lanciare la simulazione:

```bash
python -m fl_sim.cli --config configs/example_ml.toml
```

I risultati verranno scritti in `output_ml/`:

- `client_times.csv`  – tempi per round/client (training, communication, total)
- `summary.json`      – statistiche per client e metriche del modello (accuracy, macro-F1)
- `histograms.json`   – istogrammi per client.

## 4. Plottare istogrammi e metriche (con ordinamento crescente delle barre)

Lo script di plotting legge `client_times.csv` e `summary.json` e genera tre plot:

- `plot_hist_training_comm_all.png` – distribuzione globale dei tempi di training e comunicazione,
- `plot_avg_time_per_client.png`    – tempo medio per client (training + communication) con barre **ordinate in modo crescente**,
- `plot_model_metrics.png`          – accuracy e macro-F1 del modello in funzione dei round.

Per plottare usando i risultati di `output_ml/`:

```bash
python scripts/plot_results.py --dir output_ml
```

Note:

- nell’istogramma per client (`plot_avg_time_per_client.png`), i client sono ordinati in modo crescente rispetto al **tempo medio totale** (training + comunicazione), per rendere più leggibile la distribuzione.
- lo script `plot_results.py` accetta `--dir` per puntare ad altre directory di output (es. `output/` per la config di base).


> python -m fl_sim.cli --config path/to/config
> python scripts/plot_results.py --dir path/to/out
