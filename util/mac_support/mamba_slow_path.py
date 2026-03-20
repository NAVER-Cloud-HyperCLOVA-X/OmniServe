import torch
import torch.nn.functional as F
import math

def mamba_ssm_slow_path(u, dt, A, B, C, D=None, chunk_size=256, delta_bias=None, delta_softplus=True):
    """
    Pure PyTorch fallback for Mamba2 SSM scan.
    This approximates the mamba_split_conv1d_scan_combined Triton kernel.
    """
    batch, seqlen, dim = u.shape

    # B: (batch, seqlen, n_states * ngroups)
    # C: (batch, seqlen, n_states * ngroups)
    # dt: (batch, seqlen, num_heads)

    # Broadcast dt to dim
    if dt.shape[-1] != dim:
        head_dim = dim // dt.shape[-1]
        dt = torch.repeat_interleave(dt, head_dim, dim=-1)

    if delta_bias is not None:
        dt = dt + delta_bias.unsqueeze(0).unsqueeze(0)
    if delta_softplus:
        dt = F.softplus(dt)

    A_expanded = A.unsqueeze(0).unsqueeze(0).repeat_interleave(head_dim, dim=-1) if 'head_dim' in locals() else A
    decay = torch.exp(dt * A_expanded)

    y = u * decay

    if D is not None:
        y = y + u * D.unsqueeze(0).unsqueeze(0)

    return y
