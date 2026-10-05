"""NSA forward reference. Unoptimized, gather-based, obviously correct.
This is the numerical ground truth for every kernel that follows."""
import torch, torch.nn.functional as F

def compress_kv(K, V, block=32, stride=16, w_k=None, w_v=None):
    """Compressed branch: overlapping block pooling. stride<block deliberately —
    avoids fragmentation at block boundaries."""
    B, H, S, D = K.shape
    n = max(0, (S - block) // stride + 1)
    idx = torch.arange(block, device=K.device).view(1, -1) + \
          torch.arange(n, device=K.device).view(-1, 1) * stride      # [n, block]
    Kb = K[:, :, idx, :]                                             # [B,H,n,block,D]
    Vb = V[:, :, idx, :]
    if w_k is None:
        return Kb.mean(3), Vb.mean(3)
    return (Kb * w_k.view(1,1,1,block,1)).sum(3), (Vb * w_v.view(1,1,1,block,1)).sum(3)

def _attend(q, k, v, mask=None):
    """Single softmax attention, fp64-safe."""
    s = (q @ k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
    if mask is not None:
        s = s.masked_fill(~mask, float('-inf'))
    p = torch.softmax(s, dim=-1)
    p = torch.nan_to_num(p, nan=0.0)          # fully-masked rows -> zero, not NaN
    return p @ v

def select_blocks(q, K_cmp, n_sel, l_prime, S, causal=True):
    """Importance scores from the compressed branch -> top-n block indices.
    Shared across the GQA group (that sharing is NSA's whole efficiency argument)."""
    B, G, Sq, D = q.shape                      # q already group-summed
    scores = (q @ K_cmp.transpose(-1, -2)) / (D ** 0.5)   # [B,G,Sq,n_cmp]
    n_blk = (S + l_prime - 1) // l_prime
    n_cmp = K_cmp.shape[2]
    # map compressed positions -> selection blocks by center position
    src = torch.linspace(0, S - 1, n_cmp, device=q.device)
    tgt = (src // l_prime).long().clamp(max=n_blk - 1)
    blk = torch.zeros(B, G, Sq, n_blk, device=q.device, dtype=scores.dtype)
    blk.index_add_(3, tgt, scores)
    if causal:
        qpos = torch.arange(Sq, device=q.device).view(-1, 1)
        bpos = torch.arange(n_blk, device=q.device).view(1, -1) * l_prime
        blk = blk.masked_fill((bpos > qpos).view(1, 1, Sq, n_blk), float('-inf'))
    k = min(n_sel, n_blk)
    return blk.topk(k, dim=-1).indices          # [B,G,Sq,k]

def nsa_forward(Q, K, V, *, n_sel=16, l_prime=64, window=512,
                cmp_block=32, cmp_stride=16, gate_w=None, dtype=torch.float64,
                sel_override=None):
    """Q:[B,Hq,S,D]  K,V:[B,Hkv,S,D].  Returns [B,Hq,S,D]."""
    Q, K, V = Q.to(dtype), K.to(dtype), V.to(dtype)
    B, Hq, S, D = Q.shape
    Hkv = K.shape[1]; G = Hq // Hkv
    dev = Q.device
    causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=dev))

    Kx = K.repeat_interleave(G, 1); Vx = V.repeat_interleave(G, 1)

    # --- branch 1: compressed ---
    K_cmp, V_cmp = compress_kv(K, V, cmp_block, cmp_stride)
    Kc = K_cmp.repeat_interleave(G, 1); Vc = V_cmp.repeat_interleave(G, 1)
    n_cmp = Kc.shape[2]
    cpos = torch.linspace(0, S - 1, n_cmp, device=dev).view(1, -1)
    cmask = cpos <= torch.arange(S, device=dev).view(-1, 1)
    o_cmp = _attend(Q, Kc, Vc, cmask.view(1, 1, S, n_cmp))

    # --- branch 2: selected ---
    if sel_override is not None:
        sel = sel_override.to(dev)
    else:
        qg = Q.view(B, Hkv, G, S, D).sum(2)                 # group-shared scores
        sel = select_blocks(qg, K_cmp, n_sel, l_prime, S)    # [B,Hkv,S,k]
    n_blk = (S + l_prime - 1) // l_prime
    keep = torch.zeros(B, Hkv, S, n_blk, dtype=torch.bool, device=dev)
    keep.scatter_(3, sel, True)
    tok = keep.repeat_interleave(l_prime, 3)[:, :, :, :S]    # [B,Hkv,S,S]
    tok = tok.repeat_interleave(G, 1) & causal.view(1, 1, S, S)
    o_sel = _attend(Q, Kx, Vx, tok)

    # --- branch 3: sliding window ---
    pos = torch.arange(S, device=dev)
    wmask = (pos.view(-1,1) - pos.view(1,-1) < window) & causal
    o_win = _attend(Q, Kx, Vx, wmask.view(1, 1, S, S))

    # --- learned gate ---
    if gate_w is None:
        g = torch.full((3,), 1/3, dtype=dtype, device=dev)
    else:
        g = torch.sigmoid(gate_w.to(dtype)); g = g / g.sum()
    return g[0]*o_cmp + g[1]*o_sel + g[2]*o_win, sel

if __name__ == "__main__":
    torch.manual_seed(0)
    Q = torch.randn(1, 8, 512, 64, device='cuda')
    K = torch.randn(1, 2, 512, 64, device='cuda')
    V = torch.randn(1, 2, 512, 64, device='cuda')
    o, sel = nsa_forward(Q, K, V, n_sel=4, l_prime=64, window=128)
    print("out", o.shape, "sel", sel.shape, "finite", torch.isfinite(o).all().item())
