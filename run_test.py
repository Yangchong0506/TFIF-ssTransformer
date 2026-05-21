"""
TFIF-ssTransformer: Semi-Supervised Training Script
====================================================
Multimodal FeO soft sensing with Transformer + MobileViT fusion,
cosine similarity alignment, and entropy regularization.

This script uses a standard Transformer (scaled dot-product attention)
for the time-series encoder. No teacher-student distillation.

Dataset: Liuzhou Steel industrial sintering process data (real subset).
Due to proprietary constraints, only a small sample is provided for
code verification.

Usage:
    python run_test.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import gc
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

# Project modules
from transformer_mobilevit import Transformer_Model_cos_sim
from data_load import get_ss_did3_data_original, score_pre

# ── Configuration ────────────────────────────────────────────────────────

print("CUDA available:", torch.cuda.is_available())
print("PyTorch version:", torch.__version__)
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ── Hyperparameters ──────────────────────────────────────────────────────

enc_in = 17          # number of input features (time-series channels)
seq_len = 9          # input sequence length
label_len = 1        # decoder input length
pred_len = 1         # prediction horizon
moving_avg = 5       # window size for series decomposition moving average
factor = 2           # TOPK multiplier for attention
dropout = 0.05       # dropout rate
output_atten = False # whether to output attention weights
d_model = 32         # model dimension
n_heads = 4          # number of attention heads
d_ff = None          # FFN hidden dimension (None = 4*d_model)
activation = 'gelu'  # activation function
e_layers = 4         # encoder layers
d_layers = 1         # decoder layers
c_out = 1            # output dimension
delta_limit = 0.04   # tolerance for hit rate calculation
lambda_cos = 0.6      # weight for cosine similarity loss (Eq.13)
lambda_entropy = 0.15  # entropy regularization coefficient (Eq.13)

epochs = 60          # training epochs
batch_size = 10      # batch size

# ── Paths ─────────────────────────────────────────────────────────────────

data_dir = "data"
loss_dir = "loss"
results_dir = "results"

save_path = os.path.join(results_dir,
    f'student_Transformer_seq{seq_len}_best.pth')

# Create output directories
os.makedirs(loss_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# ── Load Datasets ────────────────────────────────────────────────────────

fig_training_dataset = torch.load(
    os.path.join(data_dir, f'feo_ss_3_dataset_train_len{seq_len}.pth'),
    weights_only=False)
train_loader_fig = DataLoader(
    fig_training_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
train_num = len(fig_training_dataset)

fig_validate_dataset = torch.load(
    os.path.join(data_dir, f'feo_ss_3_dataset_val_len{seq_len}.pth'),
    weights_only=False)
validate_loader_fig = DataLoader(
    fig_validate_dataset, batch_size=1, shuffle=False, num_workers=0)
val_num = len(fig_validate_dataset)

print(f"Using {train_num} images for training, {val_num} for validation.")

train_all, test_all, avg_train_y_org, std_train_y_org = get_ss_did3_data_original()


# ── Utility Functions ────────────────────────────────────────────────────

def rescale_z_score_test(data, avg, stan):
    """Reverse z-score normalization."""
    return np.array([i * stan + avg for i in data])


def weight_mse_func(y_label):
    """Create sample weights: every 4th sample has weight=1, others weight=0."""
    y_label_one = y_label.squeeze().cpu().numpy()
    weightindex = [1 if i % 4 == 0 else 0 for i in range(len(y_label_one))]
    return torch.Tensor(weightindex).unsqueeze(1).to(device)


def compute_mse(preds, targets):
    """Mean squared error (batch average)."""
    assert preds.shape == targets.shape, "Shape mismatch"
    squared_diff = (preds.cpu() - targets.cpu()) ** 2
    mse_per_sample = squared_diff.mean(dim=(1, 2))
    return mse_per_sample.mean()


# ── Training Loop ────────────────────────────────────────────────────────

for ii in range(10):  # 10 independent runs
    lr = 0.001

    # Initialize student model
    student_model = Transformer_Model_cos_sim(
        enc_in, seq_len, label_len, pred_len, moving_avg, factor, dropout,
        output_atten, d_model, n_heads, d_ff, activation,
        e_layers, d_layers, c_out
    ).to(device)
    t = student_model

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(t.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

    # Loss tracking
    losses = []
    losses_test = []
    cos_losses = []
    cos_losses_test = []
    entropy_losses = []
    entropy_losses_test = []

    best_model = None
    best_val_r2 = float("-inf")

    for e in range(epochs):
        if_save = (e % 20 == 0)

        # ── Training Phase ───────────────────────────────────────────────
        total_loss = 0.
        total_loss_cos = 0.
        total_loss_entropy = 0.

        for step, data in enumerate(train_loader_fig):
            optimizer.zero_grad()
            x_ts, X_fig, y_ts = data
            weighted_mse = weight_mse_func(y_ts)

            # Student forward
            net_out, loss_cos, entropy, enc_out = t(
                x_ts, X_fig, if_save, step, e)

            # Weighted MSE loss (semi-supervised: only 1/4 samples supervised)
            delta = net_out - y_ts.squeeze(-1)
            weighted_delta = delta * weighted_mse
            mse_train = torch.mean((weighted_delta) ** 2)

            # Batch-averaged auxiliary losses
            cos_train = torch.mean(loss_cos)
            entropy_train_loss = torch.mean(entropy)

            # Combined loss (Eq.13): L_mse + lambda_cos * L_cos - lambda_entropy * L_entropy
            loss = mse_train + lambda_cos * cos_train - lambda_entropy * entropy_train_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.detach().cpu().numpy()
            total_loss_cos += cos_train.detach().cpu().numpy()
            total_loss_entropy += entropy_train_loss.detach().cpu().numpy()

        losses.append(total_loss)
        cos_losses.append(total_loss_cos)
        entropy_losses.append(total_loss_entropy)

        print(f"[Epoch {e}] train_mse: {mse_train:.6f}, "
              f"cos_loss: {total_loss_cos:.4f}, "
              f"entropy_loss: {total_loss_entropy:.4f}")

        scheduler.step()
        lr = scheduler.get_last_lr()
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[Epoch {e}] lr: {lr[0]:.6f}")

        # ── Validation Phase ─────────────────────────────────────────────
        _ = t.eval()
        idx_prediction = []
        test_data_label = []

        total_test_loss_cos = 0.
        total_test_loss_entropy = 0.

        with torch.no_grad():
            for data_val in validate_loader_fig:
                x_ts_val, X_fig_val, y_ts_val = data_val
                if y_ts_val.abs() > 0:
                    net_out, loss_cos_test, loss_entropy_test, _ = t(
                        x_ts_val, X_fig_val)

                    test_data_label.append(y_ts_val.cpu().numpy())
                    idx_prediction.append(net_out.cpu().numpy())
                    total_test_loss_cos += loss_cos_test.detach().cpu().numpy()
                    total_test_loss_entropy += loss_entropy_test.detach().cpu().numpy()

        prediction_np = np.array(idx_prediction).reshape(-1, 1)
        label_y = np.array(test_data_label).reshape(-1, 1)

        mse_scores = score_pre(label_y, prediction_np, delta_limit=delta_limit)
        mse_value, rmse_value, r2, hr, mape = (
            mse_scores[0], mse_scores[1], mse_scores[2],
            mse_scores[3], mse_scores[4])

        losses_test.append(mse_value)
        cos_losses_test.append(total_test_loss_cos)
        entropy_losses_test.append(total_test_loss_entropy)

        print(f"[Epoch {e}] val_mse: {mse_value:.6f}, val_r2: {r2:.6f}, "
              f"val_cos: {total_test_loss_cos:.4f}, "
              f"val_entropy: {total_test_loss_entropy:.4f}")

        # Save best model
        if r2 >= best_val_r2:
            best_val_r2 = r2
            best_model = t
            torch.save(best_model.state_dict(), save_path)

        torch.cuda.empty_cache()

    # ── Save Loss Curves ─────────────────────────────────────────────────
    train_loss_arr = np.array(losses).reshape(-1, 1)
    test_loss_arr = np.array(losses_test).reshape(-1, 1)

    train_cos_arr = np.array(cos_losses).reshape(-1, 1)
    test_cos_arr = np.array(cos_losses_test).reshape(-1, 1)

    train_entropy_arr = np.array(entropy_losses).reshape(-1, 1)
    test_entropy_arr = np.array(entropy_losses_test).reshape(-1, 1)

    # Cosine loss
    cos_combined = np.hstack((train_cos_arr, test_cos_arr))
    pd.DataFrame(cos_combined,
                 columns=['train_loss_cosine', 'test_loss_cosine']).to_csv(
        os.path.join(loss_dir,
                     f'dill_weight_ss_dim6_cossim_Noentropy_did3_r0.001_'
                     f'Cossim_loss_32_32_transformer_mix_v4_seq{seq_len}_{ii}.csv'),
        index=False)

    # MSE loss
    mse_combined = np.hstack((train_loss_arr, test_loss_arr))
    pd.DataFrame(mse_combined,
                 columns=['train_loss_mse', 'test_loss_mse']).to_csv(
        os.path.join(loss_dir,
                     f'dill_weight_ss_dim6_cossim_Noentropy_did3_r_0.001_'
                     f'mse_loss_32_32_transformer_mix_v4_seq{seq_len}_{ii}.csv'),
        index=False)

    # Entropy loss
    entropy_combined = np.hstack((train_entropy_arr, test_entropy_arr))
    pd.DataFrame(entropy_combined,
                 columns=['train_loss_entropy', 'test_loss_entropy']).to_csv(
        os.path.join(loss_dir,
                     f'dill_weight_ss_dim6_cossim_Noentropy_did3_r_0.001_'
                     f'entropy_loss_32_32_transformer_mix_v4_seq{seq_len}_{ii}.csv'),
        index=False)

    # ── Final Evaluation ─────────────────────────────────────────────────
    t.load_state_dict(torch.load(save_path, weights_only=False))
    t.eval()

    idx_prediction = []
    test_data_label = []

    with torch.no_grad():
        for data_val in validate_loader_fig:
            x_ts_val, X_fig_val, y_ts_val = data_val
            if y_ts_val.abs() > 0:
                net_out, _, _, _ = t(x_ts_val, X_fig_val)
                idx_prediction.append(net_out.cpu().numpy())
                test_data_label.append(y_ts_val.cpu().numpy())

    prediction_np = np.array(idx_prediction).reshape(-1, 1)
    label_y = np.array(test_data_label).reshape(-1, 1)
    torch.cuda.empty_cache()

    mse_scores = score_pre(label_y, prediction_np, delta_limit=delta_limit)
    print(f"[Normalized] seq{seq_len} run{ii} test scores: {mse_scores}")

    # Reverse normalization for interpretable results
    label_y_test = rescale_z_score_test(
        label_y, avg_train_y_org, std_train_y_org)
    test_y_pre = rescale_z_score_test(
        prediction_np, avg_train_y_org, std_train_y_org)

    mse_scores_orig = score_pre(
        label_y_test, test_y_pre, delta_limit=delta_limit)
    print(f"[Original scale] seq{seq_len} run{ii} test scores: "
          f"MSE={mse_scores_orig[0]:.6f}, RMSE={mse_scores_orig[1]:.6f}, "
          f"R2={mse_scores_orig[2]:.6f}, HR={mse_scores_orig[3]:.4f}, "
          f"MAPE={mse_scores_orig[4]:.2f}%")

    # Save predictions
    test_results = np.hstack((test_y_pre, label_y_test))
    pd.DataFrame(test_results,
                 columns=['ss_did3_TransformerMobileMix_pre', 'test_y_label']).to_csv(
        os.path.join(results_dir,
                     f'dill_weight_ss_dim6_cossim_entropy_did3_beta{beta}_'
                     f'r0.001_Trans32_32MobileMix_pre_seq_{seq_len}_{ii}_'
                     f'rmse{mse_scores_orig[1]:.4f}_r2{mse_scores_orig[2]:.4f}.csv'),
        index=False)

print("\nTraining complete. Best R² across runs saved to results directory.")
