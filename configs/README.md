# Configurazioni del simulatore (TOML)

Questa cartella contiene i file TOML che configurano le simulazioni del simulatore DES di Federated Learning.

Ogni file TOML è composto da sezioni principali:

- `[simulation]` – parametri globali della simulazione.
- `[training_time]` – modello di tempo per il **training**.
- `[communication_time]` – modello di tempo per la **comunicazione**.
- `[output]` – dove scrivere i risultati.
- `[model]` – (opzionale) dummy model da addestrare e opzioni di misurazione del tempo.
- `[devices."<id>"]` – (opzionale) override per hardware/banda/batch size di singoli client.

Di seguito una spiegazione esaustiva di ciascuna sezione.

---

## 1. Sezione `[simulation]`

Esempio:

```toml
[simulation]
rounds = 100          # numero di round federati
num_clients = 50      # numero totale di client disponibili
clients_per_round = 10  # quanti client selezionare per round
seed = 1234           # seed RNG globale
```

Campi:

- `rounds` (int): numero di round federati.
- `num_clients` (int): numero di client totali (ID da `0` a `num_clients-1`).
- `clients_per_round` (int, opzionale): numero di client da selezionare ogni round.
  - Se assente, viene usato `participation_rate` (non mostrato qui, default 1.0).
- `participation_rate` (float, opzionale): frazione di client selezionati a ogni round (default 1.0).
- `seed` (int): seed iniziale per la simulazione.

---

## 2. Sezione `[training_time]`

Controlla come vengono generati i **tempi di training** per ciascun client.

Campo principale:

- `kind` (string): tipo di modello di tempo. Valori supportati:
  - `"lognormal"` / `"normal"` / `"exponential"` / `"constant"` – modelli parametrici semplici.
  - `"trace"` – legge tempi da un file CSV di tracce.
  - `"hardware"` – modello analitico basato su hardware + meta del modello.
  - `"ml"` – usa il modello ML addestrato in `data/ml_time_model.npz`.

Esempi:

```toml
# Esempio 1: lognormale puro
[training_time]
kind = "lognormal"
mean = 0.0
sigma = 0.5

# Esempio 2: modello ML
[training_time]
kind = "ml"
mean = 0.0      # rumore lognormale aggiuntivo (mu log-space)
sigma = 0.3
```

Campi comuni:

- `mean`, `sigma` (float): parametri base del rumore lognormale/gaussiano.
- `constant` (float): per `kind = "constant"`, tempo fisso.
- `file` / `column` (string): per `kind = "trace"`, CSV da cui leggere i tempi.

Note per `kind = "ml"`:

- richiede che esista `data/ml_time_model.npz` (generato da `scripts/train_ml_time_model.py`).
- usa come feature:
  - hardware: `cpu_cores`, `cpu_freq_ghz`, `mem_available_gb`, `accelerator_tflops`,
  - modello: `flops_per_sample`, `model_size_bytes`, `samples_per_client`,
  - training: `batch_size`, `local_epochs`.
- queste informazioni derivano da `[model]` e, per alcune, da `[devices."<id>"]`.

---

## 3. Sezione `[communication_time]`

Controlla i **tempi di comunicazione** (invio dei gradienti/modello).

Campo principale:

- `kind` (string): modello di tempo. I casi più utili qui sono:
  - `"bandwidth"` – tempo ≈ dimensione_gradiente / banda (stocastica).
  - `"lognormal"`, `"normal"`, `"constant"`, ecc. – se vuoi un modello puramente parametrico.

Esempio tipico:

```toml
[communication_time]
kind = "bandwidth"
mean = 0.0
sigma = 0.3
default_bandwidth_mean_mbps = 20.0   # banda media di default per tutti i device (Mbps)
default_bandwidth_jitter = 0.3       # jitter lognormale di default
```

Per `kind = "bandwidth"`:

- per ogni client:
  - `payload_bytes ≈ model_size_bytes` (da `[model]`);
  - `bandwidth ~ LogNormal(mu, sigma)` in Mbps;
  - `time ≈ payload_bytes / bandwidth_bps`.
- La banda si determina così:
  1. Se esiste `[devices."<id>"].bandwidth_mean_mbps` / `bandwidth_jitter`, usa quelli.
  2. Altrimenti, se definiti, usa `default_bandwidth_mean_mbps` / `default_bandwidth_jitter`.
  3. In mancanza di tutto, default interni: 10 Mbps, jitter 0.4.

---

## 4. Sezione `[output]`

Specifica dove scrivere i risultati.

```toml
[output]
directory = "output_dummy_ml"
write_traces_csv = true
write_summary_json = true
write_histograms_json = true
per_client_histograms = true
histogram_bins = 20
```

Campi:

- `directory` (string): cartella di output.
- `write_traces_csv` (bool): se `true`, scrive `client_times.csv`.
- `write_summary_json` (bool): se `true`, scrive `summary.json` (statistiche per client).
- `write_histograms_json` (bool): se `true`, scrive `histograms.json`.
- `per_client_histograms` (bool): abilita istogrammi per client.
- `histogram_bins` (int): numero di bin per istogrammi.

---

## 5. Sezione `[model]` (dummy FL model)

Controlla se e come viene addestrato un **dummy model** (logreg-like) durante la simulazione.

Esempio:

```toml
[model]
enabled = true              # se false: nessun dummy training
measure_training_time = false  # se true: usa il wall-clock del dummy training come training_time
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

Campi chiave:

- `enabled` (bool):
  - `true`  → il dummy model viene istanziato e addestrato; si ottengono metriche (accuracy, macro-F1).
  - `false` → **nessun** modello viene addestrato (anche se le info sotto vengono comunque usate dal time model ML).
- `measure_training_time` (bool):
  - `false` (default): i tempi di training vengono dal modello di tempo (`[training_time]`).
    - il dummy training, se `enabled = true`, serve solo per le metriche.
  - `true`: il simulatore misura il **wall-clock** di `local_train()` per ciascun client e usa quel valore come `training_time` (sovrascrivendo il modello di tempo).
- `flops_per_sample`, `model_size_bytes`, `samples_per_client`, `batch_size`, `local_epochs`:
  - descrivono “quanto è pesante” il modello e il training;
  - sono usati:
    - dal modello ML (`kind = "ml"`) per il tempo di training,
    - dal modello analitico `kind = "hardware"`,
    - per determinare la dimensione dei gradienti (payload) nel modello di comunicazione.

Se la sezione `[model]` manca del tutto:

- `cfg.model = None` e nessun dummy model viene creato.

---

## 6. Sezioni `[devices."<id>"]` (per-device)

Permette di specializzare hardware, GPU, banda e batch size per singoli client.

Esempio:

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
batch_size = 64         # batch locale usata nelle feature del modello ML

[devices."1"]
cpu_cores = 16
has_gpu = true
accelerator_tflops = 10.0
bandwidth_mean_mbps = 50.0
bandwidth_jitter = 0.2
batch_size = 256
```

Campi:

- `cpu_cores`, `cpu_freq_ghz`, `mem_total_gb`, `mem_available_gb`:
  - override dell’hardware rispetto all’host.
- `accelerator_tflops`:
  - TFLOPs della GPU/acceleratore per quel device.
- `has_gpu`:
  - `true` / `false`: forza presenza o assenza di GPU.
  - se non specificato, viene inferito da `accelerator_tflops > 0` o, in ultima istanza, dall’host.
- `bandwidth_mean_mbps`, `bandwidth_jitter`:
  - banda media e jitter lognormale per il modello `"bandwidth"`.
- `batch_size`:
  - batch size specifica per quel device;
  - usata nel vettore di feature del modello ML dei tempi (`log_batch_size`).

I client _non_ elencati in `[devices."<id>"]` usano:

- l’hardware dell’host (`collect_host_hardware()`),
- la banda di default (`default_bandwidth_mean_mbps` / `default_bandwidth_jitter` in `[communication_time]`),
- la `batch_size` globale definita in `[model]`.

---

## 7. Esempi di config pronti

Questa cartella contiene alcuni esempi già pronti:

- `example1_no_model_dist.toml`  
  Nessun modello, tempi da distribuzioni (training + comunicazione).

- `example2_no_model_ml.toml`  
  Nessun modello, tempi di training dal modello ML, comunicazione da modello `"bandwidth"`.

- `example3_dummy_dist.toml`  
  Dummy model addestrato, tempi di training da distribuzioni, comunicazione `"bandwidth"`.

- `example4_dummy_ml.toml`  
  Dummy model addestrato, tempi di training dal modello ML, comunicazione `"bandwidth"`.

Puoi usarli come base per comporre le tue configurazioni personalizzate.  
Per lanciare una simulazione:

```bash
python -m fl_sim.cli --config configs/example4_dummy_ml.toml
```

