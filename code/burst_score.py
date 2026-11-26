# burst_score.py -- BURST: the score with ONE transposition as the unit of length, in its shortest form (reference implementation; on the GPU server the same code is code/sx_m_tra_score.py, with the function named tra).
#
#   s(x) = d_f * B_f(x) + d_z * B_z(x),     B_g(x) = sum_{k=1..N} w_k * log( tau_g(i_k) / r_g(x, i_k) ),     w_k = (1/k) / (1 + 1/2 + ... + 1/N)
#
#   r_g(x, i)     distance between the descriptor g of the input x and of the train image x_i
#   i_1, i_2, ... the train images sorted by that distance (nearest first)
#   tau_g(i)      transposition distance of x_i: | g(x_i) - g(T x_i) |, T = the image with rows and columns swapped
#   d_g           intrinsic dimension of the descriptor, estimated from the train images
#   Higher = more in-distribution.  Nothing is trained; at inference the bank holds the N upright descriptors and the N numbers log tau.
#
#   Descriptors (computed elsewhere, one truncated forward pass of the frozen denoiser):
#     f  unit vector of [channel mean, channel std] of the read-out block U (chord distance)
#     z  amplitude profile, one number per block up to the exit block X, standardised per block with the upright train images (Euclidean distance)
#
#   usage as a module:   bank_f = build_bank(G_f, unit=True); bank_z = build_bank(G_z)      G: [N, 2, D] descriptors of the train images, index 0 = as given, index 1 = transposed
#                        s = burst(Qf, Qz, bank_f, bank_z)
#   self-check on the GPU server (val1 / val2 descriptors only; no test image):  python code/sx_m_tra_score.py <gpu>
import numpy as np, torch


def neg_log_dist(q, U, unit=False):
    """-log of the Euclidean distance of every row of q [n, D] to every row of U [N, D]  ->  [n, N]   (unit=True: rows are unit vectors, |a - b|^2 = 2 - 2 a.b)"""
    d2 = 2 - 2 * q @ U.T if unit else (q * q).sum(1, keepdim=True) + (U * U).sum(1)[None] - 2 * q @ U.T
    return -0.5 * torch.log(d2.clamp_min(1e-12)).double()


def build_bank(G, unit=False):
    """G: [N, 2, D] train descriptors (index 0 upright, index 1 transposed).  Returns (U [N, D], log tau [N], d, unit)."""
    G = torch.as_tensor(G).float(); U = G[:, 0].contiguous(); N = len(U)
    log_tau = 0.5 * ((G[:, 0] - G[:, 1]) ** 2).sum(-1).clamp_min(1e-12).log().double()                     # log | g(x_i) - g(T x_i) |
    ranks = [1, 2, 5, 10, 20]; qi = np.sort(np.random.RandomState(0).choice(N, min(5000, N), replace=False)); rows = []
    for i in range(0, len(qi), 256):                                                                       # intrinsic dimension: slope of the mean leave-one-out log distance over log k
        j = torch.as_tensor(qi[i:i + 256], device=U.device); l = neg_log_dist(U[j], U, unit); l[torch.arange(len(j), device=U.device), j] = -float('inf')
        rows.append(l.topk(max(ranks), dim=1).values[:, [r - 1 for r in ranks]])
    d = float(-1 / np.polyfit(np.log(ranks), torch.cat(rows).mean(0).cpu().numpy(), 1)[0])
    return U, log_tau, d, unit


def term(Q, U, log_tau, unit=False, bs=256):
    """B_g(x) = sum_k w_k log(tau(i_k) / r(x, i_k))  for every row of Q [n, D]  ->  [n]"""
    Q = torch.as_tensor(Q).float().to(U.device); N = len(U); w = 1.0 / torch.arange(1, N + 1, device=U.device).double(); w = w / w.sum(); out = []
    for i in range(0, len(Q), bs):
        l, ix = neg_log_dist(Q[i:i + bs], U, unit).sort(1, descending=True)                                     # nearest first
        out.append(((l + log_tau[ix]) * w[None]).sum(1))
    return torch.cat(out)


def burst(Qf, Qz, bank_f, bank_z):
    """bank_f = build_bank(G_f, unit=True) with f as unit vectors;  bank_z = build_bank(G_z) with z the standardised amplitude profile."""
    return bank_f[2] * term(Qf, bank_f[0], bank_f[1], bank_f[3]) + bank_z[2] * term(Qz, bank_z[0], bank_z[1], bank_z[3])


if __name__ == '__main__':                      # self-check against the development numbers of addendum M (val1 | val2, 5,000-image bank, noise seed 0, sigma_1 = 0.01)
    import os, sys, json, torch.nn.functional as F
    from scipy.stats import rankdata
    ROOT = os.path.expanduser('~/BURST'); dev = torch.device(f'cuda:{int(sys.argv[1])}'); REF = json.load(open(f'{ROOT}/runs/sx/m/two.json'))
    IDN = {'edmc10': 'cifar10_test', 'l6svhn': 'svhn_test', 'dpceleba': 'celeba_test', 'in64': 'in64_imagenet_test', 'fmnist': 'g32_fmnist_test'}
    def auc(x, y): r = rankdata(np.concatenate([x, y])); return 100 * float((r[:len(x)].sum() - len(x) * (len(x) + 1) / 2) / (len(x) * len(y)))
    tot = []
    for t in IDN:
        Z = np.load(f'{ROOT}/runs/sx/m/feat/sig_{t}_s010.npz'); names = [str(n) for n in Z['names']]; mu, sd = Z['BZ'][:, 0].mean(0), Z['BZ'][:, 0].std(0) + 1e-6
        bf = build_bank(F.normalize(torch.from_numpy(Z['BF']).to(dev).float(), dim=-1), unit=True); bz = build_bank(torch.from_numpy((Z['BZ'] - mu) / sd).to(dev))
        S = {n: burst(F.normalize(torch.from_numpy(Z[f'F{i}']).to(dev).float(), dim=-1), torch.from_numpy((Z[f'Z{i}'] - mu) / sd).to(dev), bf, bz).cpu().numpy() for i, n in enumerate(names)}
        idv, idf = f'devall-{IDN[t]}', f'fresh-{IDN[t]}'; sel = [n for n in names if n.startswith('sel-')]; nat = [n for n in names if n.startswith('fresh-') and n != idf and '__' not in n and 'synth' not in n]
        v1, v2 = float(np.mean([auc(S[idv], S[n]) for n in sel])), float(np.mean([auc(S[idf], S[n]) for n in nat])); tot.append((v1, v2))
        print(f'{t:9s} val1 {v1:.2f} | val2 {v2:.2f}   (development: {REF[t]["s010"]["v1"]:.2f} | {REF[t]["s010"]["v2"]:.2f})   d_f {bf[2]:.2f} d_z {bz[2]:.2f}')
        assert abs(v1 - REF[t]['s010']['v1']) < 0.02 and abs(v2 - REF[t]['s010']['v2']) < 0.02
    print(f'five-mean val1 {np.mean([a for a, _ in tot]):.2f} | val2 {np.mean([b for _, b in tot]):.2f}   SELF-CHECK OK')
