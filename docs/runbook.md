# Runbook delle campagne sperimentali

## Preparazione

Eseguire le campagne da un commit identificabile e con working tree pulito. Installare l'ambiente con `uv sync --extra gpt`; verificare quindi:

```bash
uv run python -m unittest discover -s tests -v
uv run pre-commit run --all-files
```

Ogni run produce `config.json`, `run_manifest.json`, `params.json` e `results.json`. Il manifest registra comando, commit, stato dirty, seed, hardware e versioni. I checkpoint e gli output grezzi restano esclusi da Git.

## 01 — Quantum GPT

Il confronto deve mantenere invariati corpus, preset e seed:

```bash
uv run python run.py 01 --mode compare --config fast \
  --dataset data/input.txt --seeds 42,137,256,512,1024 --name shakespeare
```

L'output aggregato contiene validation loss, perplexity, parametri e tempo. Una campagna definitiva potrà sostituire `fast` con un preset concordato, purché entrambe le varianti usino lo stesso preset.

## 02 — Quantum ViT

La pipeline ufficiale confronta patch embedding lineare e patch embedding VQC:

```bash
uv run python run.py 02 compare --dataset pathmnist --epochs 50 \
  --seeds 42 137 256 512 1024
```

Ripetere sugli eventuali dataset aggiuntivi senza modificare iperparametri tra le varianti.

## 03 — Regolarizzazione

```bash
uv run python run.py 03 ablation --epochs 50 \
  --seeds 42 137 256 512 1024 \
  --datasets pathmnist bloodmnist dermamnist
```

La campagna salva `ablation_progress.json` dopo ogni run. In caso di interruzione:

```bash
uv run python run.py 03 ablation --epochs 50 \
  --resume experiments/ablation_progress.json
```

Il confronto primario è `quantum_reg` contro `bounded_mlp` sul gap di generalizzazione.

## 04 — Kernel quantistico

```bash
uv run python run.py 04 --seed 42 --repeats 5 \
  --n-class0 80 --n-class1 20 --n-qubits 8
```

Ogni ripetizione usa un sottocampione distinto e salva gli indici selezionati, la varianza spiegata dalla PCA e le matrici dei kernel. Aumentare `--repeats` se gli intervalli risultano troppo ampi.

## Controlli prima dell'analisi

Verificare che tutte le coppie condividano seed e configurazione; che non manchino run; che nessun manifest riporti un commit inatteso; che i risultati contengano valori finiti. Le tabelle della tesi devono essere derivate dai JSON aggregati senza modificare i risultati delle singole run.
