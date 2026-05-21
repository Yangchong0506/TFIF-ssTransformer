# TFIF-ssTransformer

**T**ime-**F**requency **I**mage **F**usion with **S**emi-**S**upervised **Transformer**

Multimodal soft sensing framework for ferrous oxide (FeO) prediction in industrial sintering processes. Combines time-series process parameters with infrared thermal images via a Transformer encoder + MobileViT backbone, with cosine similarity alignment and entropy regularization for semi-supervised learning.

## Architecture

```
Time-series (17 vars) ──→ Transformer Encoder ──→ Feature Alignment ──→ Decoder ──→ FeO Prediction
                                   │                      ↑
Image (thermal frames) ──→ MobileViT ────────────────────┘
                                   │
                    Cosine Similarity + Entropy Regularization
```

- **Temporal encoder**: Standard Transformer with series decomposition and scaled dot-product attention
- **Visual encoder**: MobileViT-XXS for efficient thermal image feature extraction
- **Multimodal fusion**: Cross-modal cosine similarity alignment with entropy regularization
- **Semi-supervised**: Weighted MSE loss (labeled samples only) + cosine alignment (all samples) + entropy regularization

## Requirements

- Python >= 3.8
- PyTorch >= 1.12
- CUDA required

## Installation

```bash
git clone https://github.com/your-org/TFIF-ssTransformer.git
cd TFIF-ssTransformer
pip install -r requirements.txt
```

## Quick Start

The repository includes a small subset of real industrial data for code verification.

```bash
python run_test.py
```

The script performs 10 independent runs. Results (predictions, loss curves) are saved to `results/` and `loss/`.

To extract different subset sizes from the full dataset (if available):

```bash
python generate_sample_data.py --csv_rows 200 --train_samples 50 --val_samples 30
```

## Configuration

Key hyperparameters in `run_test.py`:

| Parameter | Default | Description |
|---|---|---|
| `enc_in` | 17 | Time-series input channels |
| `seq_len` | 9 | Input sequence length |
| `d_model` | 32 | Model hidden dimension |
| `n_heads` | 4 | Attention heads |
| `e_layers` | 4 | Encoder layers |
| `d_layers` | 1 | Decoder layers |
| `epochs` | 60 | Training epochs |
| `batch_size` | 10 | Batch size |
| `dropout` | 0.05 | Dropout rate |
| `beta` | 0.15 | Entropy regularization coefficient |
| `delta_limit` | 0.04 | Tolerance for hit-rate calculation |

## File Structure

```
TFIF-ssTransformer/
├── README.md
├── requirements.txt
├── run_test.py                          # Main training & evaluation script
├── transformer_mobilevit.py             # Core model (Transformer + MobileViT)
├── data_load.py                         # Data loading and preprocessing
├── model_config.py                      # MobileViT architecture configs
├── generate_sample_data.py              # Data subset extractor & format docs
├── layers/
│   ├── __init__.py
│   ├── Embed.py                         # Data embedding layers
│   ├── Autoformer_EncDec.py             # Encoder/Decoder with series decomposition
│   ├── SelfAttention_Family.py          # Attention mechanisms (FullAttention)
│   └── masking.py                       # Attention masks
└── data/                                # Sample real data subset
    ├── ss_did3_data_all.csv             # Time-series data (200 rows)
    ├── feo_ss_3_dataset_train_len9.pth  # Training images (50 samples)
    └── feo_ss_3_dataset_val_len9.pth    # Validation images (30 samples)
```

## Dataset

The model was developed using industrial data from a Liuzhou Steel sintering production line. Due to commercial sensitivity and intellectual property constraints, the full dataset cannot be publicly released.

A small subset of real industrial data is provided in `data/` for code verification and reproducibility validation. The provided subset is sufficient to verify that the code runs correctly and produces expected outputs.

**Data format**:
- Time-series: CSV with 17 process parameters + decoder auxiliary inputs + FeO target value
- Images: Pre-processed thermal frame tensors in PyTorch TensorDataset format (`.pth`)
- Each sample: (time_series_window [9, 19], thermal_image [1, 384, 1920], FeO_label [1, 1])

## Citation

If you use this code in your research, please cite:

```bibtex
@article{tfif-sstransformer2026,
  title={TFIF-ssTransformer: Time-Frequency Image Fusion with Semi-Supervised Transformer for FeO Soft Sensing in Sintering Processes},
  author={...},
  journal={...},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

**Note**: The code is provided for research and educational purposes. The proprietary industrial dataset is not included in full.
