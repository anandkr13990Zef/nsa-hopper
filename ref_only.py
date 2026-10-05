import torch
def ref_selected(Q,K,V,sel,l_prime,BQ,dtype=torch.float64):
    """Selected branch only, per-query-block selection. fp64 ground truth."""
    Q,K,V = Q.to(dtype),K.to(dtype),V.to(dtype)
    B,Hq,S,D = Q.shape; Hkv=K.shape[1]; G=Hq//Hkv; dev=Q.device
    NQB,KSEL = sel.shape[2], sel.shape[3]
    n_blk = (S+l_prime-1)//l_prime
    # padding (-1) routed to a scratch column so it cannot overwrite block 0
    keep = torch.zeros(B,Hkv,NQB,n_blk+1, dtype=torch.bool, device=dev)
    idx = torch.where(sel >= 0, sel, torch.full_like(sel, n_blk)).long()
    keep.scatter_(3, idx, True)
    keep = keep[..., :n_blk]
    tok = keep.repeat_interleave(l_prime,3)[:,:,:,:S]            # [B,Hkv,NQB,S]
    tok = tok.repeat_interleave(BQ,2)[:,:,:S,:]                  # [B,Hkv,S,S]
    causal = torch.tril(torch.ones(S,S,dtype=torch.bool,device=dev))
    mask = (tok & causal).repeat_interleave(G,1)
    s = (Q @ K.repeat_interleave(G,1).transpose(-1,-2)) / (D**0.5)
    s = s.masked_fill(~mask, float('-inf'))
    p = torch.nan_to_num(torch.softmax(s,-1), nan=0.0)
    return p @ V.repeat_interleave(G,1)
