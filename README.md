# Explainable Multi-Label Hate Speech Detection in Transliterated Bangla

CSE 4889 Machine Learning, United International University

Team: Md. Israfil Hossain, Md. Biplob, Diab Alam, Ifta Faisal, Md. Faiyaz Ullah Adin

We fine-tune XLM-RoBERTa on the [BanTH](https://aclanthology.org/2025.findings-naacl.403/) dataset to detect hate speech in Bangla written in English letters (Banglish), and to name which of 8 target categories apply. We explain every prediction with LIME and SHAP, and check those explanations against words marked by two human annotators.

## Main results (official BanTH test split)

| | Result |
|---|---|
| Binary macro-F1 | 76.23 (BanTH paper, XLM-R: 77.35) |
| Category macro-F1 on hate comments | 28.89 (BanTH paper, XLM-R: 29.29) |
| Annotator agreement (Cohen's kappa, word level) | 0.559 |
| SHAP token F1 vs human rationales | 0.715 (random words: 0.518) |
| Drop in P(hate) when SHAP's top words are deleted | 0.48 (random words: 0.16) |
| Harmless sentences with insult-like words flagged as hate | 26.1% (without such words: 3.2%) |

All numbers come from the files in `results/`. The full write-up is in `report/main.tex`.

## Folder structure

```
notebooks/   Colab notebooks, run in order (00 to 10)
src/         model (dual-head XLM-R) and LIME/SHAP helpers
data/        rationale set (banth_xplain_mini.csv) and bias-test sentences
results/     figures, tables (CSV + LaTeX), metrics (JSON), ASSETS.md
report/      IEEE paper (main.tex, references.bib, tables, figures)
```

## How to run

Everything runs in Google Colab (T4 GPU) with the project folder in Google Drive at `MyDrive/ML_Project`.

| Notebook | What it does |
|---|---|
| 00_setup_drive_folders | creates the folder structure |
| 01_data_exploration | downloads BanTH, checks and cleans it, saves the splits |
| 03_baseline_table | published BanTH results as baseline |
| 04_model_training | trains the dual-head model (about 20 min) |
| 05_explainability | LIME and SHAP examples |
| 06_rationale_sheet / 07_rationale_evaluation | human rationale sheets, kappa, plausibility, faithfulness |
| 08_demo_app | Gradio web app with a public link |
| 09_keyword_bias / 10_debias_training | keyword bias test and the oversampling experiment |

The trained model weights (about 1 GB) are not in this repository. Run `04_model_training` to create them, or ask the team for the Drive link.

## Data

BanTH is loaded from Hugging Face (`aplycaebous/BanTH`) and is not copied into this repository. Please cite the BanTH paper if you use it. The dataset contains real hateful and offensive comments.
