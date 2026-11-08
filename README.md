# Lobular or ductal? Dense readouts and token-level explanations

Distinguishing invasive lobular carcinoma (ILC) from invasive carcinoma of no special type (NST) on TCGA-BRCA H&E slides, with linear heads on the patch tokens of five pathology foundation models, and explaining both the predictions and the errors.

## Files

| file | content |
|---|---|
| `findings.ipynb` | the analysis and its results |
| `fetch_tiles.py` | cuts the notebook's tiles and their annotations from the local datasets and caches them |
| `reproduce.py` | trains the heads and recomputes every table and number that the notebook takes as given |
| `heads/linear_heads.pt` | the trained heads (written by `reproduce.py`) |
| `notebook_data/bank_tokens.csv` | the bank of annotated reference tokens of section 6 (written by `reproduce.py`) |
| `pyproject.toml`, `uv.lock` | the Python environment |

## Setup

```
uv sync
huggingface-cli login        # H-optimus-0, UNI2-h, Virchow2 and Hibou-B are gated: request access on their Hugging Face pages first
```

## Datasets

The notebook and `reproduce.py` read local copies of the public datasets below. Download them, then set their paths in the notebook's data cell (`DATASETS`, passed to `fetch_tiles.py`) and at the top of `reproduce.py`. The notebook needs only the first four.

| path | dataset | used by | source |
|---|---|---|---|
| `TCGA_SLIDES` | TCGA-BRCA diagnostic slides (`.svs`). The notebook needs 6 BEETLE TCGA slides (the 5 evaluation slides and A8-A0AB, the lobular host of section 8). `reproduce.py` needs the 10 BEETLE TCGA slides and the slides with a coarse tumour outline | both | NCI Genomic Data Commons |
| `BCSS` | BCSS region masks (`masks/`), region images without colour normalisation (`rgbs_raw/`) and `meta/roiBounds_BaseMagnification.csv` (Amgad et al. 2019). The repository's `download_crowdsource_dataset.py` downloads the region images at 0.25 MPP into `images/`. Rename that folder to `rgbs_raw/` | both | github.com/PathologyDataScience/BCSS |
| `NUCLS` | NuCLS dataset, `csv/` and `rgb/` (Amgad et al. 2022) | both | sites.google.com/view/nucls |
| `BEETLE` | BEETLE `annotations.zip`. `reproduce.py` also needs the Radboudumc slides `wsis/<name>.tif` | both | Zenodo record 22751764 |
| `MANIFEST` | slide table with the lobular or ductal label per patient | `reproduce.py` | provided by the panel (`slide_manifest.csv`) |
| `OUTLINES` | coarse tumour outlines of TCGA-BRCA, `coarse.zip` (Banerjee et al. 2025) | `reproduce.py` | Zenodo record 21132605 |

## Running the notebook

Set the dataset paths in the notebook's data cell (`DATASETS`), open `findings.ipynb` and run all cells. `fetch_tiles.py` cuts the 1,018 tiles the notebook uses (18 for the figures, 1,000 behind the bank of section 6) from the datasets and caches them in `data/tiles/` (1.5 GB). This takes a few minutes on the first run. A full run takes about 10 minutes on one 8 GB GPU.

## Reproducing the heads and the tables

`reproduce.py` rebuilds everything the notebook ships: the heads, the bank of section 6, the example tiles of section 5, and the tables and numbers of sections 1, 2, 3, 5, 7 and 9, which rest on about 20,000 tiles. Set the dataset paths at the top of the script (all seven, see Datasets), then run

```
uv run python reproduce.py
```

It takes about an hour and a half on one 8 GB GPU and caches tiles and tokens in `reproduction/cache/` (about 31 GB). The outputs go to `reproduction/`: copy `linear_heads.pt` to `heads/` and `bank_tokens.csv` to `notebook_data/` to use them in the notebook. `tables.md` holds the tables and the numbers quoted in the text.

The design choices are fixed below the paths: the five hospital folds, the BEETLE slides used for training and for evaluation, and the L2 penalties searched.

## Licences

The code (`findings.ipynb` as code, `fetch_tiles.py`, `reproduce.py`) is released under the MIT licence, see `LICENSE`.

The shipped artefacts, `heads/linear_heads.pt` and `notebook_data/bank_tokens.csv`, and the figures in the notebook's outputs are derived from the datasets and encoders below and are released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/): non-commercial research use, with attribution, under the same terms. They inherit the share-alike term from BEETLE, whose outlines train the lobular class and locate the bank's lobular tokens, and the non-commercial term from BEETLE and from the encoders UNI2-h, Virchow2 and Phikon-v2, on whose features part of the heads are trained.

| source | licence |
|---|---|
| [BCSS](https://github.com/PathologyDataScience/BCSS) | CC0 1.0 |
| [NuCLS](https://nucls.grand-challenge.org/License/) | CC0 1.0 |
| [BEETLE](https://zenodo.org/records/22751764) | CC BY-NC-SA 4.0 |
| TCGA-BRCA slides ([GDC](https://portal.gdc.cancer.gov/)) | open access, GDC data use terms |
| [Coarse tumour outlines](https://zenodo.org/records/21132605) (read by `reproduce.py` only) | CC BY 4.0 in the record's metadata, CC BY-NC 4.0 in its description |
| [H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0), [Hibou-B](https://huggingface.co/histai/hibou-b) | Apache-2.0, gated |
| [UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h), [Virchow2](https://huggingface.co/paige-ai/Virchow2) | CC BY-NC-ND 4.0, gated |
| [Phikon-v2](https://huggingface.co/owkin/phikon-v2) | Owkin non-commercial licence |

No data, slide, tile or model weight is redistributed here. The datasets and encoders are obtained from their sources under their own terms.
