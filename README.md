# PPI-Word

Predict protein interactions from ESM-2 fragment representations, inspect
high-scoring fragment pairs, and match explicit sequence rules to proteins.

## Installation

Use Python 3.10 or later and a PyTorch build appropriate for your CPU/CUDA device.
Clone the repository and install the package:

```bash
git clone https://github.com/name-lulangya/PPI-Word.git
cd PPI-Word
python -m pip install '.[esm,metrics]'
ppi-word --help
```

BF16 embedding and scoring require a supported GPU.

## Data and pretrained models

Datasets, pretrained scoring models and the fragment rule library are available
at [Zenodo](https://doi.org/10.5281/zenodo.24529748).

| Archive | Contents | Directory after extraction |
|---|---|---|
| `gold.tar.gz` | Human PPI benchmark | `data/gold/` |
| `cross_species.tar.gz` | Human training and cross-species benchmark | `data/cross_species/` |
| `string.tar.gz` | STRING training, evaluation and calibration data | `data/string/` |
| `structures.tar.gz` | Structural interaction data | `data/structures/` |
| `mutations.tar.gz` | Interaction mutation annotations and sequences | `data/mutations/` |
| `rule_library.tar.gz` | Fragment rules and scoring metadata | `data/rule_library/` |
| `model_weights.tar.gz` | Gold, cross-species and STRING scoring models | `model_weights/` |

Verify downloads with `sha256sum -c SHA256SUMS`. Extract the first six archives
into `data/` and the model archive into the repository directory. Each package
README describes its files, fields and sources.

Download the official [esm2_t33_650M_UR50D.pt weights](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt)
and place the file in `weights/`.

## Quick start

The examples below use the two chains from [PDB 4JW2](https://www.rcsb.org/structure/4JW2)
in `examples/4jw2/`. Commands write
to new output directories. `embed` creates `index.json`, which subsequent
commands read through `--embeddings`.

### Predict a PPI

```bash
ppi-word embed --fasta examples/4jw2/sequences.fasta \
  --esm-checkpoint weights/esm2_t33_650M_UR50D.pt --device cuda:0 \
  --mode fps64 --precision bf16 --residue-storage-dtype float32 \
  --storage-dtype float16 --output outputs/embeddings

ppi-word predict --embeddings outputs/embeddings/index.json \
  --pairs examples/4jw2/pairs.csv --checkpoint model_weights/string.pt \
  --config configs/string.json --device cuda:0 --precision bf16 --output outputs/prediction
```

`predictions.csv` contains the logit and sigmoid probability for each pair.

### Inspect high-scoring fragment pairs

```bash
ppi-word embed --fasta examples/4jw2/sequences.fasta \
  --esm-checkpoint weights/esm2_t33_650M_UR50D.pt --device cuda:0 \
  --mode stride1 --precision bf16 --residue-storage-dtype float16 \
  --pooling-method prefix_sum --storage-dtype float32 --output outputs/stride1

ppi-word local --embeddings outputs/stride1/index.json \
  --pairs examples/4jw2/pairs.csv --checkpoint model_weights/string.pt \
  --config configs/string.json --device cuda:0 --precision bf16 --top-k 50 --output outputs/local
```

The output lists fragment sequences, positions and local model scores.
For the full biological datasets, use the same embedding settings and these inputs:

| Dataset | FASTA and pair table | Additional arguments |
|---|---|---|
| Structures | `data/structures/sequences.fasta`, `data/structures/complexes.csv` | `embed --max-sequence-length 3905`; `local`/`rules --pair-id-column complex_id` |
| Mutations | `data/mutations/sequences.fasta`, `data/mutations/pairs.csv` | `embed --max-sequence-length 3526` |

### Predict with rules

```bash
ppi-word rules --fasta examples/4jw2/sequences.fasta \
  --pairs examples/4jw2/pairs.csv --rules data/rule_library/rules.parquet \
  --metadata data/rule_library/metadata.json --memory-limit 4GB --threads 2 \
  --output outputs/rules
```

`predictions.csv` contains rule scores for all input pairs, including pairs with
no matches. `matches.csv` reports the matched rules, positions and contributions.
Rule scores are not probabilities; their definition is in the rule package README.

## Training and rule extraction

Task configurations are in `configs/`. Use the task-specific embedding settings:

| Task | Window mode | ESM-2 precision | Residue storage | Fragment storage |
|---|---|---|---|---|
| Gold and cross-species prediction | `no_overlap` | `fp32` | `float16` | `float16` |
| STRING prediction | `fps64` | `bf16` | `float32` | `float16` |
| STRING rule construction | `stride1` | `bf16` | `float32` | `float16` |

<details>
<summary>Training and evaluation commands</summary>

```bash
ppi-word embed --fasta data/gold/sequences.fasta \
  --esm-checkpoint weights/esm2_t33_650M_UR50D.pt --device cuda:0 \
  --mode no_overlap --precision fp32 --residue-storage-dtype float16 \
  --storage-dtype float16 --output outputs/gold_embeddings

torchrun --standalone --nproc_per_node=4 -m ppi_word.cli.main train \
  --embeddings outputs/gold_embeddings/index.json \
  --train data/gold/train.csv --valid data/gold/valid.csv \
  --config configs/gold.json --device cuda:0 --output outputs/gold_training

ppi-word predict --embeddings outputs/gold_embeddings/index.json \
  --pairs data/gold/test.csv --checkpoint outputs/gold_training/best.pt \
  --config configs/gold.json --device cuda:0 --precision fp32 --output outputs/gold_prediction

ppi-word evaluate --pairs data/gold/test.csv \
  --predictions outputs/gold_prediction/predictions.csv \
  --score-column probability --threshold 0.5 --output outputs/gold_metrics
```

For cross-species and STRING, prepare the corresponding embeddings using the
settings above, then train with:

```bash
torchrun --standalone --nproc_per_node=8 -m ppi_word.cli.main train \
  --embeddings outputs/cross_embeddings/index.json \
  --train data/cross_species/human_train.csv --valid data/cross_species/human_test.csv \
  --config configs/cross_species.json --device cuda:0 --output outputs/cross_training

torchrun --standalone --nproc_per_node=4 -m ppi_word.cli.main train \
  --embeddings outputs/string_embeddings/index.json \
  --train data/string/train.csv --valid data/string/valid.csv \
  --config configs/string.json --data-manifest data/string/manifest.json \
  --device cuda:0 --output outputs/string_training
```

Keep the training process count consistent with the task configuration. `best.pt` can be
used directly for prediction. Use FP32 prediction for cross-species and BF16
for STRING. The evaluator reports AUROC and AP (labelled AUPRC in the paper),
plus precision, recall and F1 when a threshold is supplied. Published thresholds
are in `configs/evaluation_thresholds.json`.

</details>

<details>
<summary>Rule construction commands</summary>

Prepare complete STRING stride-1 embeddings, then calibrate the selection threshold:

```bash
ppi-word embed --fasta data/string/sequences.fasta \
  --pairs data/string/train.csv data/string/calibration_pairs.csv \
  --esm-checkpoint weights/esm2_t33_650M_UR50D.pt --device cuda:0 \
  --mode stride1 --precision bf16 --residue-storage-dtype float32 \
  --storage-dtype float16 --output outputs/string_stride1

ppi-word calibrate --pairs data/string/calibration_pairs.csv \
  --embeddings outputs/string_stride1/index.json --checkpoint model_weights/string.pt \
  --config configs/string.json --device cuda:0 --precision bf16 \
  --tail-probability 1e-7 --output outputs/calibration
```

The published threshold and selection bounds are supplied in
`configs/rule_extraction.json`. The following commands use that fixed policy.
Complete all extraction shards before cataloguing and all negative-support
shards before assembly:

```bash
for i in {0..31}; do
  ppi-word extract --pairs data/string/train.csv \
    --embeddings outputs/string_stride1/index.json --checkpoint model_weights/string.pt \
    --config configs/string.json --policy configs/rule_extraction.json \
    --device cuda:0 --precision bf16 --shards 32 --shard-id "$i" \
    --output "outputs/events/part$i"
done

ppi-word rule-catalog --events outputs/events/part* \
  --memory-limit 16GB --threads 4 --output outputs/catalog

for i in {0..31}; do
  ppi-word negative-support --catalog outputs/catalog/candidates.parquet \
    --fasta data/string/sequences.fasta --pairs data/string/train.csv \
    --shards 32 --shard-id "$i" --batch-pairs 100 \
    --memory-limit 16GB --threads 4 --output "outputs/negative/part$i"
done

ppi-word rule-assemble --catalog outputs/catalog/candidates.parquet \
  --supports outputs/negative/part* --release-id my-rules-1 \
  --memory-limit 16GB --threads 4 --output outputs/my_rules
```

The 32 shards divide the workload; memory, thread and query-batch settings bound
resource use. Adjust these to your resources while keeping each shard set
complete. Match the new library using `rules` with its `rules.parquet` and
`metadata.json`.

</details>

For large embedding runs, use `--shards N --shard-id I` with separate output
directories. `--pairs` limits embedding to proteins in the specified pair tables;
`--resume` continues an interrupted embedding shard. Merge completed shards with:

```bash
ppi-word merge-embeddings --indexes outputs/embeddings/shard*/index.json \
  --output outputs/embeddings/index.json
```

## Attribution

PPI-Word code is available under the [MIT License](LICENSE).
Source attributions are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
