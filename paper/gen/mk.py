# mk.py -- numbers, table bodies and the data figures of the paper (method: BURST = the score with the transposed unit), written from data/*.json (copied from the GPU server; md5 checked below).
#   Server names of the image splits: test2 = test, testc = test'.  "ours" in sum_tra.json = the score with the transposed unit (server row o:transp).
#   data: sum_tra.json, sum_tra_pairs2.json (code/sx_r_sum.py: stored test scores re-tabulated with the transposed unit as the reference row; values rounded to one decimal, half up),
#         fig_tra.json (sx_r_fig.py), devslim.json (sx_r_dev.py / sx_r_devsum.py / sx_r_slim.py: val1 | val2, bank of 5,000 train images), mv2.json (sx_r_mv.py: train images),
#         eqsum.json (sx_r_eq.py), blkslim.json (sx_r_blk.py), blkrule.json (the block rule with the transposition alone), fullval.json (full bank on val1 | val2),
#         bench.json and sum_pass.json (time; the pass is the same for every unit of length)
#         summary_m.json (sx_m_sum.py: val1 | val2 with the noise level of the pass changed, and with units made from colour transformations)
#   writes ../numbers.tex, ../tab_*.tex, ../supp_*.tex, ../figs/fig_hist_in64.tex, fig_hist_c10.tex, fig_pairs.tex, fig_blk.tex, fig_rho.tex, fig_sym.tex
import json, os, math, hashlib
from decimal import Decimal, ROUND_HALF_UP
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.dirname(HERE); D = os.path.join(OUT, 'data'); J = lambda n: json.load(open(f'{D}/{n}.json'))
for fn, m in (('sum_tra', 'e60668e3484e219f31e7dad15435ebee'), ('sum_tra_pairs2', '5948dc463086965c536c92c7ec1aac97'), ('fig_tra', 'e9a5a7a28777671d481891286b0343d9'), ('devslim', '7152b252148be642345dc15287e02904'),
              ('mv2', '40561d3a2d4eb38a545e2e28b87ef0b9'), ('eqsum', 'd5ccc1bd7af1eccbb621d232a9323d6c'), ('blkslim', 'e3eb9874f26bd2400c2e2e8721bb84db'), ('fullval', '61ee2017c6bcbb1295d17b141ce8297c'), ('summary_m', 'edd1c3d259f31b89495114d9f9b5463d')):
    assert hashlib.md5(open(f'{D}/{fn}.json', 'rb').read()).hexdigest() == m, fn
S = J('sum_tra'); P = J('sum_tra_pairs2'); FD = J('fig_tra'); DV = J('devslim'); MV2 = J('mv2'); EQ = J('eqsum'); BLK = J('blkslim'); SK = J('fullval'); SM = J('summary_m'); B = J('bench'); PASS = J('sum_pass')
TAGS = ['edmc10', 'l6svhn', 'dpceleba', 'in64', 'fmnist']
NAME = {'edmc10': 'CIFAR-10', 'l6svhn': 'SVHN', 'dpceleba': 'CelebA', 'in64': 'ImageNet-64', 'fmnist': 'FashionMNIST'}; SID = {'edmc10': 'Cten', 'l6svhn': 'Svhn', 'dpceleba': 'Celeb', 'in64': 'In', 'fmnist': 'Fm'}
IDD = {'edmc10': 'cifar10_test', 'l6svhn': 'svhn_test', 'dpceleba': 'celeba_test', 'in64': 'in64_imagenet_test', 'fmnist': 'g32_fmnist_test'}
NAT = {'edmc10': ['svhn_test', 'textures', 'celeba_test', 'tin_val', 'lsun_fix', 'imagenet_fix', 'imagenet32_val', 'places365_val', 'cifar100_test'],
       'l6svhn': ['cifar10_test', 'cifar100_test', 'celeba_test', 'textures', 'tin_val', 'lsun_fix', 'imagenet_fix', 'imagenet32_val', 'places365_val'],
       'dpceleba': ['cifar10_test', 'cifar100_test', 'svhn_test', 'textures', 'tin_val', 'lsun_fix', 'imagenet_fix', 'imagenet32_val', 'places365_val'],
       'in64': ['in64_ssb_hard', 'in64_ninco', 'in64_inaturalist', 'in64_textures', 'in64_openimage_o', 'in64_places365', 'in64_celeba_test'],
       'fmnist': ['g32_mnist_test', 'g32_kmnist_test', 'g32_omniglot_eval', 'g32_notmnist']}
SETN = {'svhn_test': 'SVHN', 'textures': 'Textures', 'celeba_test': 'CelebA', 'tin_val': 'Tiny-ImageNet', 'lsun_fix': 'LSUN', 'imagenet_fix': 'ImageNet',
        'imagenet32_val': 'ImageNet32', 'places365_val': 'Places365', 'cifar100_test': 'CIFAR-100', 'cifar10_test': 'CIFAR-10',
        'in64_ssb_hard': 'SSB-hard', 'in64_ninco': 'NINCO', 'in64_inaturalist': 'iNaturalist', 'in64_textures': 'Textures', 'in64_openimage_o': 'OpenImage-O',
        'in64_places365': 'Places365', 'in64_celeba_test': 'CelebA', 'g32_mnist_test': 'MNIST', 'g32_kmnist_test': 'KMNIST', 'g32_omniglot_eval': 'Omniglot',
        'g32_notmnist': 'notMNIST', 'synth_noise': '一様ノイズ', 'synth_const': '定数画像', 'g32_synth_noise': '一様ノイズ', 'g32_synth_const': '定数画像'}
PRIOR = ['NLL', 'IC-png', 'IC-j2k', 'TT', 'MSMA*', 'DiffPath', 'GEPC', 'EigenScore', 'CFS', 'CFS-kNN']
LABEL = {'NLL': 'NLL(尤度)', 'IC-png': 'IC(PNG)', 'IC-j2k': 'IC(JPEG2000)', 'TT': 'TT', 'MSMA*': 'MSMA型', 'DiffPath': 'DiffPath', 'GEPC': 'GEPC', 'EigenScore': 'EigenScore',
         'RotCls4': '参考：回転の予測(ResNet-18)', 'CFS': 'CFS', 'CFS-kNN': 'CFS(kNN ヘッド)'}
SHORT = {'NLL': 'NLL', 'IC-png': 'IC-PNG', 'IC-j2k': 'IC-J2K', 'TT': 'TT', 'MSMA*': 'MSMA型', 'DiffPath': 'DiffPath', 'GEPC': 'GEPC', 'EigenScore': 'EigenScore', 'CFS': 'CFS', 'CFS-kNN': 'CFS-kNN'}
CITE = {'NLL': 'song2021score', 'IC-png': 'serra2020input', 'IC-j2k': 'serra2020input', 'TT': 'nalisnick2019typ', 'MSMA*': 'mahmood2021msma', 'DiffPath': 'heng2024diffpath',
        'GEPC': 'alexis2026gepc', 'EigenScore': 'shoushtari2026eigenscore', 'RotCls4': 'hendrycks2019ss', 'CFS': 'rouzoumka2026cfs', 'CFS-kNN': 'rouzoumka2026cfs'}
PASSES = {'NLL': '128+128', 'IC-png': '128+128', 'IC-j2k': '128+128', 'TT': '128+128', 'MSMA*': '20', 'DiffPath': '10', 'GEPC': '14', 'EigenScore': '180', 'RotCls4': '4', 'CFS': '1', 'CFS-kNN': '1'}

def rd(v, n=1): return str(Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-n), rounding=ROUND_HALF_UP))
def sg(v, n=1): s = rd(abs(v), n); return s if float(s) == 0 else ('+' if v > 0 else '-') + s
f2 = lambda v: rd(v, 2); mean = lambda v: sum(v) / len(v)
W = {}; NUM = {}
def num(name, val): assert name.isalpha() and name not in NUM, name; NUM[name] = val
T = S['test2']['rows']; O = S['test2']['ours']; TC = S['testc']['rows']; OC = S['testc']['ours']

# ---------------------------------------------------------------- numbers of the text: test and test'
for q, o, R in (('T', O, T), ('C', OC, TC)):
    num(f'Ours{q}', rd(o['mean'])); num(f'OursLo{q}', rd(o['ci'][0])); num(f'OursHi{q}', rd(o['ci'][1])); num(f'Ud{q}', rd(o['vflip_mean'])); num(f'Hf{q}', rd(o['hflip_mean'])); num(f'Fpr{q}', rd(o['fpr_mean']))
    for nm, r in (('Prior', 'CFS-kNN'), ('Diag', 'CFS'), ('Trained', 'A trained'), ('NoUnit', 'no orbit'), ('Seven', 'all7'), ('Local', 'nn10'), ('LocalH', 'nnh'), ('UHf', 'o:hflip'), ('UVf', 'o:vflip'), ('URot', 'o:rot180'), ('UCfour', 'o:C4'), ('USwap', 'o:swap')):
        x = R[r]; num(f'{nm}{q}', rd(x['mean'])); num(f'{nm}Ud{q}', rd(x['vflip_mean'])); num(f'{nm}Fpr{q}', rd(x['fpr_mean'])); num(f'{nm}Hf{q}', rd(x['hflip_mean']))
        num(f'Gap{nm}{q}', rd(abs(x['d5'][0]))); num(f'Gap{nm}Lo{q}', rd(x['d5'][1])); num(f'Gap{nm}Hi{q}', rd(x['d5'][2])); w = x['wtl']; num(f'{nm}Win{q}', str(w[0])); num(f'{nm}Tie{q}', str(w[1])); num(f'{nm}Loss{q}', str(w[2]))
    for i, t in enumerate(TAGS):
        num(f'Ours{SID[t]}{q}', rd(o['ids'][i])); num(f'Trained{SID[t]}{q}', rd(R['A trained']['ids'][i])); num(f'Ud{SID[t]}{q}', rd(o['vflip'][i])); num(f'Hf{SID[t]}{q}', rd(o['hflip'][i])); num(f'NoUnit{SID[t]}{q}', rd(R['no orbit']['ids'][i]))
assert T['all7']['d5'][1] > 0 and T['all7']['mean'] < O['mean'] and T['no orbit']['mean'] < O['mean'] and T['o:hflip']['mean'] < T['no orbit']['mean']
num('RotT', rd(T['RotCls4']['mean'])); num('RotUdT', rd(T['RotCls4']['vflip_mean'])); num('EigenUdT', rd(T['EigenScore']['vflip_mean'])); num('PriorFprBestT', rd(min(T[r]['fpr_mean'] for r in PRIOR))); num('GepcT', rd(T['GEPC']['mean']))
for i, t in enumerate(TAGS): num(f'PriorBest{SID[t]}T', rd(max(T[r]['ids'][i] for r in PRIOR)))
num('DiffPathSvhnT', rd(T['DiffPath']['ids'][1])); num('DiffPathInT', rd(T['DiffPath']['ids'][3])); num('NllFmT', rd(T['NLL']['ids'][4])); num('UVfSvhnT', rd(T['o:vflip']['ids'][1])); num('UHfFmT', rd(T['o:hflip']['ids'][4]))
assert O['fpr_mean'] < 0.5 * min(T[r]['fpr_mean'] for r in PRIOR)
LOSS = [(t, o, e['b']) for t in TAGS for o, e in P[t].items() if o != '__hflip' and e['b'][2] in ('l', 'x')]; num('NLossPairs', str(len(LOSS))); num('NLossSvhn', str(sum(1 for t, o, b in LOSS if t == 'l6svhn')))
num('CfsKnnCelebInT', rd(P['in64']['in64_celeba_test']['k'][0])); assert P['in64']['in64_celeba_test']['k'][0] < 50   # the kNN head of CFS ranks CelebA as more in-distribution than ImageNet-64
for nm, (t, o) in (('PairCHund', ('edmc10', 'cifar100_test')), ('PairSsb', ('in64', 'in64_ssb_hard')), ('PairNinco', ('in64', 'in64_ninco')), ('PairInat', ('in64', 'in64_inaturalist'))): num(f'{nm}T', rd(P[t][o]['o']))
# time (GPU, ms per image, batch 128): the pass and the search do not depend on which transformation gave the unit
sr = lambda t: B[t]['f'] + B[t]['z']
num('PassX', rd(mean([PASS[t]['X'] for t in TAGS]), 2)); num('PassXMin', rd(min(PASS[t]['X'] for t in TAGS), 2)); num('PassXMax', rd(max(PASS[t]['X'] for t in TAGS), 2)); num('AllX', rd(mean([B[t]['ours_X'] / B[t]['full'] for t in TAGS]), 2))
num('AllKnn', rd(mean([B[t]['knn_full'] / B[t]['full'] for t in TAGS]), 2)); num('SearchShareMax', rd(100 * max(sr(t) / B[t]['ours_X'] for t in TAGS), 0)); num('SearchShareMin', rd(100 * min(sr(t) / B[t]['ours_X'] for t in TAGS), 0))
num('DimFMin', rd(min(S['dims'][t]['f'] for t in TAGS), 0)); num('DimFMax', rd(max(S['dims'][t]['f'] for t in TAGS), 0)); num('DimZMin', rd(min(S['dims'][t]['z'] for t in TAGS), 0)); num('DimZMax', rd(max(S['dims'][t]['z'] for t in TAGS), 0))
num('NzMin', str(min(S['blocks'][t][2] + 1 for t in TAGS))); num('NzMax', str(max(S['blocks'][t][2] + 1 for t in TAGS)))

# ---------------------------------------------------------------- Table 1: comparison on test
COLS = [('ids', 0), ('ids', 1), ('ids', 2), ('ids', 3), ('ids', 4), ('mean', None), ('vflip_mean', None), ('fpr_mean', None)]
def col(r, c): x = (O if r is None else T[r])[c[0]]; return float(x if c[1] is None else x[c[1]])
best = {c: (min if c[0] == 'fpr_mean' else max)(col(r, c) for r in PRIOR) for c in COLS}
def cell(r, c, mark):
    v = col(r, c); s = rd(v)
    if r is None: lower = c[0] == 'fpr_mean'; return f'\\cellcolor{{firstblue}}\\textbf{{{s}}}' if ((v <= best[c]) if lower else (v >= best[c])) else s
    return s + ('$^{\\dagger}$' if mark and v == best[c] else '')
cmp_ = lambda r: '/'.join(str(x) for x in T[r]['cmp'])
L = ['% written by gen/mk.py: AUROC (%) on test; "cmp": pairs (of 51) where Ours is higher by 0.05 or more / within 0.05 / lower']
for r in PRIOR: L.append(f'{LABEL[r]}~\\cite{{{CITE[r]}}} & ' + ' & '.join(cell(r, c, True) for c in COLS) + f' & {cmp_(r)} & {PASSES[r]}\\\\')
L.append('\\midrule'); r = 'RotCls4'; L.append(f'{LABEL[r]}~\\cite{{{CITE[r]}}} & ' + ' & '.join(cell(r, c, False) for c in COLS) + f' & {cmp_(r)} & {PASSES[r]}\\\\')
L.append('\\midrule'); L.append('\\textbf{Ours：\\method} & ' + ' & '.join(cell(None, c, False) for c in COLS) + f' & -- & {NUM["PassX"]}$^{{\\ddagger}}$\\\\\\bottomrule'); W['tab_main.tex'] = '\n'.join(L) + '\n'
w = T['CFS-kNN']['cmp']; num('PriorUpT', str(w[0])); num('PriorSameT', str(w[1])); num('PriorDownT', str(w[2]))

# ---------------------------------------------------------------- the symmetries of the square: train-only structure and each of them as the unit
ORI = MV2['ori']; ix = ORI.index
JN = {'rot90': '90度回転', 'rot180': '180度回転', 'rot270': '270度回転', 'hflip': '左右反転', 'vflip': '上下反転', 'transp': '転置', 'atransp': '反転置'}
ORD = ['transp', 'rot270', 'atransp', 'rot90', 'rot180', 'vflip', 'hflip']
CLS = {'hflip': '$G_p$', 'rot180': '$G_p\\Rm^{2}$', 'vflip': '$G_p\\Rm^{2}$', 'rot90': '$G_p\\Rm$', 'atransp': '$G_p\\Rm$', 'rot270': '$G_p\\Rm^{3}$', 'transp': '$G_p\\Rm^{3}$'}
PARTNER = {'rot90': 'atransp', 'atransp': 'rot90', 'rot270': 'transp', 'transp': 'rot270', 'rot180': 'vflip', 'vflip': 'rot180'}; SWAP = ['rot90', 'rot270', 'transp', 'atransp']; KEEP = ['rot180', 'hflip', 'vflip']; NATID = ['edmc10', 'dpceleba', 'in64']
auc_ = lambda t, k: MV2['res'][t]['ori']['auroc'][ix(k)]; gm = lambda t, g, a, b='transp': math.exp(MV2['res'][t][g]['mlog'][ix(a)] - MV2['res'][t][g]['mlog'][ix(b)])
tr = [auc_(t, 'transp') for t in TAGS]; num('AucTrMin', rd(min(tr))); num('AucTrMax', rd(max(tr))); assert min(tr) > 76
for t in TAGS: num(f'AucTr{SID[t]}', rd(auc_(t, 'transp'))); num(f'AucHf{SID[t]}', rd(auc_(t, 'hflip')))
num('AucSwapMin', rd(min(auc_(t, k) for t in TAGS for k in SWAP))); num('AucKeepSvhnMin', rd(min(auc_('l6svhn', k) for k in KEEP))); num('AucKeepSvhnMax', rd(max(auc_('l6svhn', k) for k in KEEP)))
assert all(auc_(t, 'hflip') < 54 for t in NATID) and max(auc_('l6svhn', k) for k in KEEP) < 65
for k in SWAP: assert min(auc_(t, k) for t in TAGS) > 76, k
for k in KEEP: assert min(auc_(t, k) for t in TAGS) < 65, k                       # every other symmetry fails on at least one ID data set
nrow = 0; win = []
for t in NATID:
    for g in 'fz':
        R_ = MV2['res'][t][g]['rho']
        for k, p in PARTNER.items():
            r_ = {o: R_[ix(k)][ix(o)] for o in ORI if o != k}; assert max(r_, key=r_.get) == p, (t, g, k); nrow += 1; win.append(r_[p])
num('RhoRows', str(nrow)); num('RhoInMin', f2(min(win))); num('RhoInMax', f2(max(win)))
sw = [gm(t, g, k) for t in TAGS for g in 'fz' for k in SWAP if k != 'transp']; num('SwapRatioMin', f2(min(sw))); num('SwapRatioMax', f2(max(sw))); assert 0.96 < min(sw) and max(sw) < 1.05, (min(sw), max(sw))
rs = [MV2['res'][t][g]['rho'][ix('transp')][ix(k)] for t in TAGS for g in 'fz' for k in SWAP if k != 'transp']; num('SwapRhoMin', f2(min(rs))); num('SwapRhoMax', f2(max(rs)))
dv = lambda k, c: DV['res'][k + '|fz'][c]; two = lambda k, c: DV['two'][k][c]; TST = {'hflip': 'o:hflip', 'vflip': 'o:vflip', 'rot180': 'o:rot180'}
L = ['% written by gen/mk.py from data/mv2.json (train images), data/devslim.json (val1 | val2, bank of 5,000 train images) and data/sum_tra.json (test)']
for k in ORD:
    b0, b1 = ('\\textbf{', '}') if k == 'transp' else ('', ''); tv = rd(O['mean']) if k == 'transp' else (rd(T[TST[k]]['mean']) if k in TST else '--')
    L.append(f"{b0}{JN[k]}{'(\\method)' if k == 'transp' else ''}{b1} & {CLS[k]} & " + ' & '.join(rd(auc_(t, k)) for t in TAGS) + f" & {b0}{f2(dv('e:' + k, 'v1'))}{b1} & {b0}{f2(dv('e:' + k, 'v2'))}{b1} & ${sg(two('e:' + k, 'zT'), 2)}$ & {b0}{tv}{b1}\\\\" + ('\\addlinespace[1.5pt]' if k in ('rot270', 'rot90', 'vflip') else ''))
L.append('\\midrule')
for lab, kk, tv in (('7つすべて(幾何平均)', 'D4', T['all7']['mean']), ('軸を入れ替える4つ(幾何平均)', 'side', T['o:swap']['mean']), ('3つの回転(幾何平均)', 'rot', T['o:C4']['mean'])):
    L.append(f"\\multicolumn{{7}}{{@{{}}l}}{{{lab}}} & {f2(dv(kk, 'v1'))} & {f2(dv(kk, 'v2'))} & ${sg(two(kk, 'zT'), 2)}$ & {rd(tv)}\\\\")
L.append(f"\\multicolumn{{7}}{{@{{}}l}}{{単位なし(距離のみ)}} & {f2(dv('none', 'v1'))} & {f2(dv('none', 'v2'))} & -- & {rd(T['no orbit']['mean'])}\\\\\\bottomrule"); W['tab_unit.tex'] = '\n'.join(L) + '\n'
n1, n2 = dv('none', 'v1'), dv('none', 'v2')
assert dv('e:hflip', 'v1') < n1 and dv('e:hflip', 'v2') < n2 and all(dv('e:' + k, 'v1') > n1 and dv('e:' + k, 'v2') > n2 for k in SWAP)
assert dv('D4', 'v1') <= dv('e:transp', 'v1') and dv('D4', 'v2') <= dv('e:transp', 'v2') and dv('side', 'v2') <= dv('e:transp', 'v2') + 0.05
for nm, k in (('None', 'none'), ('Tr', 'e:transp'), ('Seven', 'D4'), ('Hf', 'e:hflip'), ('Vf', 'e:vflip'), ('Rot', 'e:rot180'), ('Shuf', 'shuf'), ('Nn', 'nn10'), ('Qswap', 't:qswap'), ('Hshift', 't:hshift'), ('Neg', 't:neg'),
               ('Fonly', None), ('Zonly', None), ('NoD', 'nod'), ('Rone', 'R1'), ('Rten', 'R10'), ('Rhund', 'R100'), ('Rthou', 'R1000'), ('ExpTwoOne', 'exp21'), ('ExpOneTwo', 'exp12'), ('UnitOnly', 'unit only')):
    r = DV['res']['e:transp|f'] if nm == 'Fonly' else (DV['res']['e:transp|z'] if nm == 'Zonly' else DV['res'][k + '|fz']); num(f'Val{nm}One', f2(r['v1'])); num(f'Val{nm}Two', f2(r['v2']))
num('ValNoneFpr', rd(dv('none', 'f2'))); num('ValTrFpr', rd(dv('e:transp', 'f2')))
v1s = [dv('e:' + k, 'v1') for k in SWAP]; v2s = [dv('e:' + k, 'v2') for k in SWAP]; num('ValSwapOneMin', f2(min(v1s))); num('ValSwapOneMax', f2(max(v1s))); num('ValSwapTwoMin', f2(min(v2s))); num('ValSwapTwoMax', f2(max(v2s)))
num('ZtTr', f2(two('e:transp', 'zT'))); num('ZA', f2(two('e:transp', 'zA'))); num('VrTr', f2(two('e:transp', 'vr'))); num('ZtHf', sg(two('e:hflip', 'zT'), 2)); num('ZtShuf', sg(two('shuf', 'zT'), 2)); num('ZtNn', sg(two('nn10', 'zT'), 2)); num('CorrNn', sg(two('nn10', 'corr'), 2)); num('CorrTr', sg(two('e:transp', 'corr'), 2))
bn_gap = max(abs(two(k, 'bnS') - dv(k, 'v2')) for k in DV['two']); num('BnGapMax', rd(math.ceil(bn_gap * 10 - 1e-9) / 10, 1)); assert two('e:transp', 'zS') > two('e:transp', 'zA') and two('shuf', 'zS') < two('shuf', 'zA') and two('e:hflip', 'zS') < two('e:hflip', 'zA')
# controls (val1 | val2)
CTRL = [('\\textbf{転置(\\method)}', 'e:transp', 'o', 'o'), ('4分割の入れ替え', 't:qswap', 'o', 'x'), ('横に半分ずらす', 't:hshift', 'o', 'x'), ('符号の反転', 't:neg', 'x', '--'), ('$2{\\times}2$ の平均', 't:blur', 'x', '--'), ('中央の拡大', 't:zoom', 'x', '--'),
        ('灰色化', 't:grey', 'x', '--'), ('別のノイズ', 't:noise', 'x', '--')]
MK = {'o': '$\\checkmark$', 'x': '$\\times$', '--': '--'}
crow_ = lambda lab, k, a, b: f"{lab} & {MK[a]} & {MK[b]} & {f2(dv(k, 'v1'))} & {f2(dv(k, 'v2'))} & {f2(two(k, 'bnS'))} & ${sg(two(k, 'zT'), 2)}$ & {f2(two(k, 'vr'))}\\\\"
L = ['% written by gen/mk.py from data/devslim.json (val1 | val2; bank of 5,000 train images)'] + [crow_(*c) for c in CTRL] + ['\\midrule', crow_('転置の単位を別の train 画像に', 'shuf', '--', '--'), crow_('10番目の近傍の距離', 'nn10', '--', '--'),
     f"単位なし(距離のみ) & -- & -- & {f2(n1)} & {f2(n2)} & {f2(two('e:transp', 'bnA'))} & -- & 1\\\\\\bottomrule"]
C5 = SM['col5']; assert abs(C5['tr|fz'][1] - dv('e:transp', 'v2')) < 0.006 and abs(C5['none|fz'][1] - n2) < 0.006                    # the colour runs reproduce the reference rows
colrow = lambda lab, k, a: f"{lab} & {MK['x']} & {MK['--']} & {f2(C5[k][0])} & {f2(C5[k][1])} & -- & -- & --\\\\"            # a colour map changes the value of every pixel: not a permutation of the pixels
i_ = L.index('\\midrule'); L[i_:i_] = ['\\addlinespace[1.5pt]', colrow('色：チャネルの巡回', 'gbr|fz', 'o'), colrow('色：チャネルの5通りの入れ替え', 'col|fz', 'o'), colrow('色：色相の180度回転', 'h180|fz', 'x')]
W['tab_ctrl.tex'] = '\n'.join(L) + '\n'
for nm, k in (('ColAll', 'col|fz'), ('ColCyc', 'gbr|fz'), ('ColHue', 'h180|fz'), ('ColF', 'col|f'), ('ColZ', 'col|z')): num(nm + 'One', f2(C5[k][0])); num(nm + 'Two', f2(C5[k][1]))
assert C5['col|fz'][0] < n1 and C5['col|fz'][1] < n2 and C5['col|f'][1] < C5['none|f'][1] - 10 and abs(C5['col|z'][1] - C5['none|z'][1]) < 0.5
nonperm = ['t:blur', 't:zoom', 't:grey', 't:noise']; assert all(dv(k, 'v1') < n1 and dv(k, 'v2') < n2 for k in nonperm) and dv('shuf', 'v1') < n1 and dv('shuf', 'v2') < n2 and dv('nn10', 'v1') < n1 and dv('nn10', 'v2') < n2
assert all(n2 < dv(k, 'v2') < dv('e:transp', 'v2') and dv(k, 'v1') < n1 for k in ['t:qswap', 't:hshift', 't:neg'])
num('ValNonPermTwoMin', f2(min(dv(k, 'v2') for k in nonperm))); num('ValNonPermTwoMax', f2(max(dv(k, 'v2') for k in nonperm)))
# the parts of the score with the transposed unit (val1 | val2)
def prow(lab, r, last=''): return f"{lab} & {f2(r['v1'])} & {f2(r['v2'])} & {rd(r['f2'])} & {rd(r['ud'])}\\\\{last}"
R_ = DV['res']
W['tab_parts.tex'] = '\n'.join(['% written by gen/mk.py from data/devslim.json (val1 | val2; bank of 5,000 train images; five-data-set means)', prow('\\textbf{\\method}', R_['e:transp|fz'], '\\midrule'),
    prow('単位なし(距離のみ)', R_['none|fz']), prow('単位のみ(距離なし)', R_['unit only|fz'], '\\midrule'), prow('割合 $2:1$\\,($\\tau/r^{2}$)', R_['exp21|fz']), prow('割合 $1:2$\\,($\\tau^{2}/r$)', R_['exp12|fz'], '\\midrule'),
    prow('最も近い1枚だけ', R_['R1|fz']), prow('近い10枚の平均', R_['R10|fz']), prow('近い100枚の平均', R_['R100|fz']), prow('近い1000枚の平均', R_['R1000|fz'], '\\midrule'),
    prow('$f$ だけ', R_['e:transp|f']), prow('$z$ だけ', R_['e:transp|z']), prow('$d_g$ を掛けない', R_['nod|fz'], '\\bottomrule')]) + '\n'
assert R_['e:transp|f']['v2'] < R_['e:transp|fz']['v2'] and R_['e:transp|z']['v2'] < R_['e:transp|fz']['v2'] and R_['R1|fz']['v2'] < R_['e:transp|fz']['v2'] and R_['nod|fz']['v2'] < R_['e:transp|fz']['v2']
idg = lambda k, i: DV['res'][k + '|fz']['id2'][i]
for nm, i in (('Cten', 0), ('Svhn', 1), ('In', 3)): g_ = [idg('e:' + k, i) - idg('none', i) for k in SWAP]; assert min(g_) > 1.5; num(f'SymidSwapGain{nm}Min', rd(math.floor(min(g_) * 10 + 1e-6) / 10))
assert all(idg('e:' + k, 2) == idg('none', 2) == 100.0 for k in SWAP); fd_ = [idg('none', 4) - idg('e:' + k, 4) for k in SWAP]; assert 0 < min(fd_) and max(fd_) < 0.2; num('SymidFmDropMin', f2(min(fd_))); num('SymidFmDropMax', f2(max(fd_)))
assert idg('e:vflip', 1) < idg('none', 1) and idg('e:hflip', 1) < idg('none', 1)
assert R_['R100|fz']['v1'] > R_['e:transp|fz']['v1'] and max(R_[k + '|fz']['v2'] for k in ('R1', 'R10', 'R100', 'R1000')) < R_['e:transp|fz']['v2']
assert R_['exp12|fz']['v1'] < R_['e:transp|fz']['v1'] and R_['exp12|fz']['v2'] < R_['e:transp|fz']['v2'] and R_['exp21|fz']['v1'] > R_['e:transp|fz']['v1'] and R_['exp21|fz']['v2'] < R_['e:transp|fz']['v2']
num('GainFonly', f2(R_['e:transp|fz']['v2'] - R_['e:transp|f']['v2'])); num('ValNoneFOne', f2(R_['none|f']['v1'])); num('ValNoneFTwo', f2(R_['none|f']['v2'])); num('ValNoneZOne', f2(R_['none|z']['v1'])); num('ValNoneZTwo', f2(R_['none|z']['v2']))
# full bank on val (addendum K): the transposed unit, no unit, seven copies, the input's own transposed copy (two passes)
F_ = SK
for nm, k in (('FullTr', 'tr'), ('FullNone', 'none'), ('FullSeven', 'seven'), ('Own', 'own')): num(f'Val{nm}One', f2(F_[k]['v1'])); num(f'Val{nm}Two', f2(F_[k]['v2']))
assert F_['own']['v1'] < F_['tr']['v1'] and F_['own']['v2'] < F_['tr']['v2'] and F_['seven']['v2'] < F_['tr']['v2']

# ---------------------------------------------------------------- the output of the trained denoiser (train images)
r25 = lambda t: dict(zip(ORI, EQ['res'][t]['sig']['0.2512']['rel'])); r01 = lambda t: dict(zip(ORI, EQ['res'][t]['sig']['0.01']['rel'])); EQT = [t for t in TAGS if t != 'l6svhn']
assert all(min(r25(t), key=r25(t).get) == 'hflip' for t in EQT) and all(r25(t)['transp'] > 1.25 * r25(t)['hflip'] for t in TAGS)
num('EqTrMin', rd(min(r25(t)['transp'] for t in TAGS), 3)); num('EqTrMax', rd(max(r25(t)['transp'] for t in TAGS), 3)); num('EqHfMin', rd(min(r25(t)['hflip'] for t in EQT), 3)); num('EqHfMax', rd(max(r25(t)['hflip'] for t in EQT), 3))
num('EqTrHfMin', f2(min(r25(t)['transp'] / r25(t)['hflip'] for t in TAGS))); num('EqTrHfMax', f2(max(r25(t)['transp'] / r25(t)['hflip'] for t in TAGS)))
num('EqLowTrHfMin', f2(min(r01(t)['transp'] / r01(t)['hflip'] for t in TAGS))); num('EqLowTrHfMax', f2(max(r01(t)['transp'] / r01(t)['hflip'] for t in TAGS))); assert all(r01(t)['transp'] > r01(t)['hflip'] for t in TAGS)
L = ['% written by gen/mk.py from data/eqsum.json (2,000 train images per ID data set; full precision)']
for k in ORD: L.append(f"{JN[k]} & " + ' & '.join(rd(r25(t)[k], 3) for t in TAGS) + ' & ' + ' & '.join(rd(r01(t)[k], 3) for t in TAGS) + '\\\\' + ('\\addlinespace[1.5pt]' if k in ('rot270', 'rot90', 'vflip') else ''))
W['supp_eq.tex'] = '\n'.join(L) + '\\bottomrule\n'
L = ['% written by gen/mk.py from data/mv2.json and data/devslim.json (val2; bank of 5,000 train images)']
for k in ORD: L.append(f"{JN[k]} & " + ' & '.join(f"{rd(auc_(t, k))} & {f2(DV['res']['e:' + k + '|fz']['id2'][i])}" for i, t in enumerate(TAGS)) + '\\\\' + ('\\addlinespace[1.5pt]' if k in ('rot270', 'rot90', 'vflip') else ''))
L.append('\\midrule'); L.append('単位なし(距離のみ) & ' + ' & '.join(f"-- & {f2(DV['res']['none|fz']['id2'][i])}" for i in range(5)) + '\\\\\\bottomrule'); W['supp_symid.tex'] = '\n'.join(L) + '\n'

# ---------------------------------------------------------------- supplement: configuration, time, test', every pair
MODEL = {'edmc10': ('EDM(VP, DDPM++)~\\cite{karras2022edm}', 34, 'デコーダ $8{\\times}8$'), 'l6svhn': ('VP-SDE~\\cite{kamkari2024geometric}', 13, 'デコーダ $8{\\times}8$'),
         'dpceleba': ('iDDPM~\\cite{nichol2021improved}', 33, 'デコーダ $8{\\times}8$'), 'in64': ('EDM2-XS(条件なし)~\\cite{karras2024edm2}', 37, 'デコーダ $8{\\times}8$'), 'fmnist': ('U-Net(EDMの前処理), 本研究で学習', 29, 'エンコーダ $8{\\times}8$')}
NTR = {'edmc10': '5万', 'l6svhn': '5万', 'dpceleba': '5万', 'in64': '1万', 'fmnist': '5万'}; RES = {'edmc10': '$32{\\times}32$', 'l6svhn': '$32{\\times}32$', 'dpceleba': '$32{\\times}32$', 'in64': '$64{\\times}64$', 'fmnist': '$32{\\times}32$'}
L = ['% written by gen/mk.py']
for t in TAGS:
    m, nb, part = MODEL[t]; u, a, x = S['blocks'][t]; d = S['dims'][t]; assert BLK[t]['iu'] == u and BLK[t]['ix'] == x and len(BLK[t]['lab']) == nb
    L.append(f"{NAME[t]} & {RES[t]} & {m} & {NTR[t]} & {nb} & {u + 1}({part}) & {rd(auc_(t, 'transp'))} & {x + 1} & {rd(d['f'])} & {rd(d['z'])}\\\\" + ('\\bottomrule' if t == TAGS[-1] else ''))
W['supp_models.tex'] = '\n'.join(L) + '\n'
BR = J('blkrule')                                                                         # full-precision record of the block rule with the transposition alone (data/blkrule.json)
for t in TAGS: assert BR[t]['iu'] == BLK[t]['iu'] and rd(BR[t]['tr_iu']) == rd(BLK[t]['tr'][BLK[t]['iu']]) and rd(BR[t]['tr_max']) == rd(max(BLK[t]['tr'])), t
same_u = [t for t in TAGS if BR[t]['tr_argmax'] == BR[t]['iu']]; num('NSameU', str(len(same_u))); assert [t for t in TAGS if t not in same_u] == ['dpceleba'] and abs(BR['dpceleba']['tr_argmax'] - BR['dpceleba']['iu']) == 1
num('CelebTrAtU', rd(BR['dpceleba']['tr_iu'], 2)); num('CelebTrBest', rd(BR['dpceleba']['tr_max'], 2))
L = ['% written by gen/mk.py: GPU, batch 128, ms per image']
for t in TAGS:
    b = B[t]; L.append(f"{NAME[t]} & {rd(b['full'], 2)} & {rd(b['to_X'], 2)} & {rd(b['f'], 3)} & {rd(b['z'], 3)} & {rd(b['ours_X'], 2)} & {rd(PASS[t]['X'], 2)} & {rd(b['ours_X'] / b['full'], 2)}\\\\" + ('\\bottomrule' if t == TAGS[-1] else ''))
W['supp_time.tex'] = '\n'.join(L) + '\n'
def trow(R, o, lab, r, last=''):
    x = o if r is None else R[r]; g = lambda f: ('--' if x[f] is None else rd(x[f])); return f"{lab} & {g('mean')} & {g('vflip_mean')} & {g('hflip_mean')} & {g('fpr_mean')} & {'--' if r is None else '/'.join(str(v) for v in x['wtl'])}\\\\{last}"
ROWSC = [('\\textbf{\\method}', None, '\\midrule'), ('単位なし(距離のみ)', 'no orbit', ''), ('単位：10番目の近傍の距離', 'nn10', ''), ('単位：近傍の平均距離($w_k$)', 'nnh', '\\midrule'), ('単位：左右反転', 'o:hflip', ''), ('単位：上下反転', 'o:vflip', ''), ('単位：180度回転', 'o:rot180', ''),
         ('単位：軸を入れ替える4つ', 'o:swap', ''), ('単位：3つの回転', 'o:C4', ''), ('単位：4つの鏡映', 'o:flips', ''), ('単位：7つすべて', 'all7', '\\midrule'), ('CFS(kNN ヘッド)~\\cite{rouzoumka2026cfs}', 'CFS-kNN', ''), ('CFS(対角ガウス)~\\cite{rouzoumka2026cfs}', 'CFS', ''), ('参考(学習あり)', 'A trained', '\\bottomrule')]
def trow2(lab, r, last=''):
    out = []
    for R, o in ((T, O), (TC, OC)):
        x = o if r is None else R[r]; out += [rd(x['mean']), rd(x['vflip_mean']), rd(x['hflip_mean']), rd(x['fpr_mean']), '--' if r is None else '/'.join(str(v) for v in x['wtl'])]
    return f"{lab} & " + ' & '.join(out[:5]) + ' & & ' + ' & '.join(out[5:]) + f'\\\\{last}'
W['supp_test2.tex'] = '\n'.join(["% written by gen/mk.py: rows scored on test and on test' (five-data-set means)"] + [trow2(*r) for r in ROWSC]) + '\n'

def cc(x, v): return ('\\cellcolor{firstblue}' if v in ('w', 'h') else ('\\cellcolor{lossred}' if v in ('l', 'x') else '')) + rd(x)
L = ['% written by gen/mk.py']
for t in TAGS:
    order = NAT[t] + [k for k in P[t] if 'synth' in k] + [IDD[t] + '__vflip', '__hflip']
    for j, o in enumerate(order):
        e = P[t][o]; name = '上下反転' if o.endswith('__vflip') else ('左右反転$^{\\S}$' if o == '__hflip' else SETN[o])
        cells = [rd(e['o']), cc(*e['n']), cc(*e['k']), (cc(e['b'][0], e['b'][2]) + f"\\,{{\\tiny {SHORT[e['b'][1]]}}}") if 'b' in e else '--']
        L.append((f'\\multirow{{{len(order)}}}{{*}}{{{NAME[t]}}}' if j == 0 else '') + f' & {name} & ' + ' & '.join(cells) + '\\\\')
    L.append('\\bottomrule' if t == TAGS[-1] else '\\midrule')
W['supp_pairs_t.tex'] = '\n'.join(L) + '\n'

# ---------------------------------------------------------------- Figure: score histograms without / with the unit (stored test scores, noise seed 0; two bins merged: width 0.5)
def steps(h):
    lo, hi, n = FD['hist'][h]['edges']; w = (hi - lo) / n; out = {}
    for key in ('id', 'ood'):
        v = FD['hist'][h][key]; out[key] = [(lo + 2 * w * j, (v[2 * j] + v[2 * j + 1]) / 2) for j in range(n // 2)]
    keep = [j for j in range(n // 2) if out['id'][j][1] > 0 or out['ood'][j][1] > 0]; a_, b_ = max(min(keep) - 1, 0), min(max(keep) + 2, n // 2)
    st = lambda p: ' '.join(f'({x:.2f},{y:.4f})' for x, y in p[a_:b_]) + f' ({p[b_ - 1][0] + 2 * w:.2f},0)'
    return st(out['id']), st(out['ood']), FD['hist'][h]['auroc']
def hist_fig(tag, width, height, legname):
    L = ['% written by gen/mk.py from data/fig_tra.json (stored test scores, noise seed 0)', '\\begin{tikzpicture}', f'\\begin{{groupplot}}[group style={{group size=2 by 1, horizontal sep=7mm}}, width={width}, height={height},',
         '  tick label style={font=\\figlab}, label style={font=\\figlab}, title style={font=\\figlab, yshift=-1.6mm, align=center}, xlabel={標準化したスコア},',
         '  legend style={font=\\figlab, draw=none, fill=none, legend columns=2, /tikz/every even column/.append style={column sep=3mm}}]']
    for j, (k, ttl) in enumerate(((f'{tag}|logr', '距離のみ'), (f'{tag}|tra', '\\method'))):
        a_, b_, au = steps(k); L.append(f"\\nextgroupplot[title={{{ttl}\\\\AUROC {rd(au)}\\%}}" + (f', ylabel={{密度}}, legend to name={legname}' if j == 0 else '') + ']')
        L.append(f'\\addplot[const plot, navyc, line width=0.7pt] coordinates {{{a_}}};'); L.append(f'\\addplot[const plot, redc, densely dashed, line width=0.7pt] coordinates {{{b_}}};')
        if j == 0: L.append('\\legend{分布内の画像, 分布外の画像}')
    return '\n'.join(L + ['\\end{groupplot}', '\\end{tikzpicture}\\\\[-0.5mm]', f'\\pgfplotslegendfromname{{{legname}}}']) + '\n'
W['figs/fig_hist_in64.tex'] = hist_fig('in64', '38mm', '33mm', 'histlegA'); W['figs/fig_hist_c10.tex'] = hist_fig('edmc10', '56mm', '42mm', 'histlegB')
for nm, k in (('HistInDist', 'in64|logr'), ('HistInOurs', 'in64|tra'), ('HistCtenDist', 'edmc10|logr'), ('HistCtenOurs', 'edmc10|tra')): num(nm, rd(FD['hist'][k]['auroc']))

# ---------------------------------------------------------------- Figure: every pair, Ours against CFS (kNN head) and against the best of the ten prior methods (test)
MARK = {'edmc10': ('*', 'navyc'), 'l6svhn': ('square*', 'redc'), 'dpceleba': ('triangle*', 'black'), 'in64': ('diamond*', 'navyc!55'), 'fmnist': ('pentagon*', 'black!45')}
L = ['% written by gen/mk.py from data/sum_tra_pairs2.json (the 51 pairs of the test images; AUROC, %)', '\\begin{tikzpicture}',
     '\\begin{groupplot}[group style={group size=2 by 1, horizontal sep=10mm}, width=45mm, height=45mm, tick label style={font=\\figlab}, label style={font=\\figlab}, title style={font=\\figlab, yshift=-1.6mm},',
     '  legend style={font=\\figlab, draw=none, fill=none, legend columns=5, /tikz/every even column/.append style={column sep=1.5mm}}]']
for j, (col_, ttl, xl) in enumerate((('k', '(a) CFS の kNN ヘッドとの比較', 'CFS(kNN ヘッド)'), ('b', '(b) 10手法の組ごとの最良との比較', '10手法の最良'))):
    allv = [v for t in TAGS for o, e in P[t].items() if o != '__hflip' for v in (e['o'], e[col_][0])]; lo_, hi_ = min(allv), max(allv)
    L.append(f"\\nextgroupplot[title={{{ttl}}}, xlabel={{{xl}}}" + (', ylabel={\\method}, legend to name=pairleg' if j == 0 else '') + ']')
    L.append(f'\\addplot[black!45, line width=0.4pt, forget plot] coordinates {{({lo_},{lo_}) ({hi_},{hi_})}};')
    for t in TAGS:
        mk_, c_ = MARK[t]; pts_ = ' '.join(f"({e[col_][0]},{e['o']})" for o, e in P[t].items() if o != '__hflip')
        L.append(f'\\addplot[only marks, mark={mk_}, mark size=1.15pt, {c_}] coordinates {{{pts_}}};')
    if j == 0: L.append('\\legend{' + ', '.join(NAME[t] for t in TAGS) + '}')
L += ['\\end{groupplot}', '\\end{tikzpicture}\\\\[-0.5mm]', '\\pgfplotslegendfromname{pairleg}']; W['figs/fig_pairs.tex'] = '\n'.join(L) + '\n'

# ---------------------------------------------------------------- Figure: per block, how well the copies under three symmetries are told from the upright train images (train images only)
L = ['% written by gen/mk.py from data/blkslim.json (5,000 train images per ID data set, noise seed 0; maximum over the neighbour rank R)', '\\begin{tikzpicture}',
     '\\begin{groupplot}[group style={group size=5 by 1, horizontal sep=7.5mm}, width=41mm, height=33mm, tick label style={font=\\figlab}, title style={font=\\scriptsize, yshift=-1.5mm, text height=1.6ex, text depth=0.3ex},',
     '  xlabel style={font=\\figlab}, ylabel style={font=\\figlab}, xlabel={ブロック(入力側 $\\rightarrow$ 出力側)}, every axis plot/.append style={line width=0.7pt},',
     '  legend style={font=\\figlab, draw=none, fill=none, legend columns=3, /tikz/every even column/.append style={column sep=3mm}}]']
cs = lambda v: ' '.join(f'({i + 1},{x})' for i, x in enumerate(v))
for j, t in enumerate(TAGS):
    b = BLK[t]; L.append(f"\\nextgroupplot[title={{{NAME[t]}}}" + (', ylabel={見分けやすさ(AUROC, \\%)}, legend to name=blkleg' if j == 0 else '') + ']')
    L.append(f"\\draw[black!55, densely dotted, line width=0.6pt] (axis cs:{b['iu'] + 1},\\pgfkeysvalueof{{/pgfplots/ymin}}) -- (axis cs:{b['iu'] + 1},\\pgfkeysvalueof{{/pgfplots/ymax}}) node[pos=0.08, right, inner sep=1pt, font=\\figlab, text=black!75] {{U}};")
    L.append(f"\\addplot[navyc] coordinates {{{cs(b['tr'])}}};"); L.append(f"\\addplot[black, densely dashed] coordinates {{{cs(b['r2'])}}};"); L.append(f"\\addplot[redc] coordinates {{{cs(b['hf'])}}};")
    if j == 0: L.append('\\legend{転置, 180度回転, 左右反転}')
L += ['\\end{groupplot}', '\\end{tikzpicture}\\\\[-0.5mm]', '\\pgfplotslegendfromname{blkleg}']; W['figs/fig_blk.tex'] = '\n'.join(L) + '\n'

# ---------------------------------------------------------------- Figure (supplement): rank correlation between the log displacements of every two symmetries (mean of the three natural-image IDs)
ORD2 = ['hflip', 'rot180', 'vflip', 'rot90', 'atransp', 'rot270', 'transp']
L = ['% written by gen/mk.py from data/mv2.json (train images; mean over CIFAR-10, CelebA and ImageNet-64 of the Spearman correlation)', '\\begin{tikzpicture}[x=6.3mm, y=6.3mm, font=\\figlab]']
for p, g in enumerate('fz'):
    x0 = p * 9.6; M_ = [[sum(MV2['res'][t][g]['rho'][ix(a)][ix(b)] for t in NATID) / 3 for b in ORD2] for a in ORD2]; L.append(f"\\node[anchor=north, font=\\scriptsize] at ({x0 + 3.5},-0.25) {{記述子 ${g}$}};")
    for i, a in enumerate(ORD2):
        L.append(f"\\node[anchor=east] at ({x0 - 0.05},{6.5 - i}) {{{JN[a]}}};"); L.append(f"\\node[anchor=south west, rotate=45, inner sep=1pt] at ({x0 + i + 0.35},7.05) {{{JN[a]}}};")
        for j2, b in enumerate(ORD2):
            v = M_[i][j2]; sh = int(round(100 * max(0.0, min(1.0, v)))); L.append(f"\\fill[navyc!{sh}] ({x0 + j2},{6 - i}) rectangle ({x0 + j2 + 1},{7 - i}); \\node[text={'white' if sh > 55 else 'black'}] at ({x0 + j2 + 0.5},{6.5 - i}) {{{'1' if a == b else '$' + sg(v, 2).lstrip('+') + '$'}}};")
    L.append(f"\\draw[black!60, line width=0.3pt] ({x0},0) rectangle ({x0 + 7},7);")
    for c0, c1 in ((1, 3), (3, 5), (5, 7)): L.append(f"\\draw[redc, line width=0.9pt] ({x0 + c0},{7 - c1}) rectangle ({x0 + c1},{7 - c0});")
W['figs/fig_rho.tex'] = '\n'.join(L + ['\\end{tikzpicture}']) + '\n'

# ---------------------------------------------------------------- Figure: (a) the eight symmetries by coset, the transposition marked; lowest train-only AUROC over the five ID data sets.  (b), (c): static schematics
CMX = {'id': '1,0,0,1', 'hflip': '-1,0,0,1', 'rot180': '-1,0,0,-1', 'vflip': '1,0,0,-1', 'rot90': '0,1,-1,0', 'atransp': '0,1,1,0', 'rot270': '0,-1,1,0', 'transp': '0,-1,-1,0'}      # (x, y) -> (a x + c y, b x + d y), y upwards
JN2 = dict(JN, id='そのまま'); GRP = [(['id', 'hflip'], '分布内にとどまる', 'black!70'), (['rot180', 'vflip'], '上下が逆', 'black!70'), (['rot90', 'atransp'], '横倒し', 'navyc'), (['rot270', 'transp'], '横倒し', 'navyc')]
L = ['% written by gen/mk.py: (a) the eight symmetries of the square by coset of {identity, hflip}; numbers: data/mv2.json (lowest of the five ID data sets, train images).  (b), (c): schematics.',
     '\\begin{tikzpicture}[x=1mm, y=1mm, font=\\figlab, line cap=round, line join=round, >={Stealth[length=1.6mm, width=1.3mm]}]',
     '\\node[anchor=south west, font=\\scriptsize, inner sep=0pt] at (-1,22.5) {\\textbf{(a)}\\ 正方形の8つの対称変換の中の転置};']
x = 5.0
for els, ttl, col in GRP:
    xa = x
    for k in els:
        L.append(f'\\begin{{scope}}[shift={{({x},9)}}]\\draw[{"redc, line width=0.9pt" if k == "transp" else "black!45, line width=0.4pt"}] (-4.3,-4.3) rectangle (4.3,4.3);\\begin{{scope}}[cm={{{CMX[k]},(0,0)}}]'
                 '\\draw[black, line width=0.9pt] (-3.4,-3.1) -- (3.4,-3.1);\\draw[black, line width=0.7pt] (-1.2,-3.1) -- (-1.2,3.1);\\fill[redc] (-1.2,3.1) -- (2.5,2.0) -- (-1.2,0.9) -- cycle;\\end{scope}\\end{scope}')
        L.append(f"\\node[anchor=north, inner sep=1pt{', text=redc' if k == 'transp' else ''}] at ({x},4.1) {{{JN2[k]}}};")
        L.append(f"\\node[anchor=north, inner sep=1pt] at ({x},0.6) {{{'--' if k == 'id' else rd(min(auc_(t, k) for t in TAGS))}}};"); x += 10.2
    L.append(f'\\draw[{col}, line width=0.5pt] ({xa - 4.3},15.6) -- ({xa - 4.3},16.6) -- ({x - 10.2 + 4.3},16.6) -- ({x - 10.2 + 4.3},15.6);'); L.append(f'\\node[anchor=south, text={col}, inner sep=1.2pt] at ({(xa + x - 10.2) / 2},16.6) {{{ttl}}};'); x += 2.6
L.append('\\node[anchor=north west, inner sep=1pt, align=left, text width=88mm] at (-1,-3.2) {下の数：その変換をした train 画像を正立の train 画像と見分ける AUROC(\\%)の, 5つの分布内での最小. 横倒しにする4つだけが, どの分布内でも見分けられる.};')
xb = x + 2.5
L += [f'\\begin{{scope}}[shift={{({xb},0)}}]', '\\node[anchor=south west, font=\\scriptsize, inner sep=0pt] at (-1,22.5) {\\textbf{(b)}\\ 記述子の2つの成分};',
      '\\coordinate (G) at (4,3); \\coordinate (GT) at (30,15); \\coordinate (E) at (17,9);',
      '\\draw[black!55, line width=0.5pt] (G) -- (GT); \\draw[->, navyc, line width=0.9pt] (E) -- (G); \\fill[navyc] (G) circle (0.75); \\draw[navyc, fill=white, line width=0.6pt] ($(GT)+(-0.7,-0.7)$) rectangle ++(1.4,1.4); \\fill[black] (E) circle (0.5);',
      '\\node[anchor=north, text=navyc, inner sep=2pt] at ($(G)+(0,-0.6)$) {$g(x)$}; \\node[anchor=south, text=navyc, inner sep=2pt] at ($(GT)+(0,0.8)$) {$g(\\Tm x)$}; \\node[anchor=south east, inner sep=1.5pt] at ($(E)+(0.2,0.5)$) {$e(x)$};',
      '\\node[anchor=north west, text=navyc, inner sep=1pt] at ($(G)!0.5!(E)+(0.4,-0.6)$) {$o(x)$};',
      '\\draw[black, line width=0.4pt, |<->|] ($(G)+(-2.2,4.77)$) -- node[above, sloped, inner sep=1.5pt] {$\\tau=2|o(x)|$} ($(GT)+(-2.2,4.77)$);',
      '\\node[anchor=north west, inner sep=1pt, align=left, text width=36mm] at (-1,-3.2) {$e$：転置で変わらない成分.\\\\$o$：転置で符号が変わる成分.};', '\\end{scope}']
xc = xb + 41
L += [f'\\begin{{scope}}[shift={{({xc},0)}}]', '\\node[anchor=south west, font=\\scriptsize, inner sep=0pt] at (-1,22.5) {\\textbf{(c)}\\ 定理1：ずれは対数密度の勾配の差};',
      '\\foreach \\a/\\b in {8/5, 5.2/3.2, 2.5/1.5}{\\draw[navyc, line width=0.5pt, rotate around={20:(7.5,1.5)}] (7.5,1.5) ellipse (\\a mm and \\b mm);}',
      '\\foreach \\a/\\b in {8/5, 5.2/3.2, 2.5/1.5}{\\draw[redc, densely dashed, line width=0.5pt, rotate around={70:(33,1.5)}] (33,1.5) ellipse (\\a mm and \\b mm);}',
      '\\node[text=navyc, anchor=north, inner sep=1pt] at (7.5,-4.6) {正立の分布}; \\node[text=redc, anchor=north, inner sep=1pt] at (33,-6.9) {転置した分布};',
      '\\coordinate (Y) at (20,20.4); \\coordinate (A) at (14.6,13.2); \\coordinate (Bq) at (25.4,13.2);',
      '\\draw[->, navyc, line width=0.8pt] (Y) -- (A); \\draw[->, redc, line width=0.8pt] (Y) -- (Bq); \\draw[->, black, line width=1.0pt] (A) -- (Bq); \\fill[black] (Y) circle (0.6);',
      '\\node[anchor=west, inner sep=2pt] at ($(Y)+(0.6,0.4)$) {入力 $y$}; \\node[anchor=east, text=navyc, inner sep=1.5pt] at ($(A)+(-0.6,0)$) {$D(y)$}; \\node[anchor=west, text=redc, inner sep=1.5pt] at ($(Bq)+(0.6,0)$) {$\\Tm D(\\Tm y)$};',
      '\\node[anchor=north, inner sep=2pt] at (20,12.6) {ずれ $\\Delta(y)$};', '\\end{scope}', '\\end{tikzpicture}']
W['figs/fig_sym.tex'] = '\n'.join(L) + '\n'

VS = ['s002', 's005', 's010', 's020', 's050']; L = ['% written by gen/mk.py from data/summary_m.json (val2 AUROC per noise level; below: transposed copies told apart, train images only)']
for t in TAGS:
    r = SM['sig'][t]; bv = max(VS, key=lambda v: r[v][1]) if max(r[v][1] for v in VS) - min(r[v][1] for v in VS) > 0.02 else None; L.append(f'{NAME[t]} & ' + ' & '.join((f'\\textbf{{{f2(r[v][1])}}}' if v == bv else f2(r[v][1])) for v in VS) + '\\\\')
L.append('\\midrule'); L.append('5つの平均 val1 & ' + ' & '.join(f2(SM['sig5'][v][0]) for v in VS) + '\\\\'); L.append('5つの平均 val2 & ' + ' & '.join((f'\\textbf{{{f2(SM["sig5"][v][1])}}}' if v == 's010' else f2(SM['sig5'][v][1])) for v in VS) + '\\\\\\midrule')
L.append('\\multicolumn{6}{@{}l}{\\textit{転置した train 画像の見分けやすさ}}\\\\')
for t in TAGS: L.append(f'{NAME[t]} & ' + ' & '.join(rd(SM['sig'][t][v][3]) for v in VS) + '\\\\' + ('\\bottomrule' if t == TAGS[-1] else ''))
W['supp_sig.tex'] = '\n'.join(L) + '\n'; assert max(VS, key=lambda v: SM['sig5'][v][1]) == 's010' and all(SM['sig'][t]['s050'][3] >= SM['sig'][t]['s002'][3] for t in TAGS) and abs(SM['sig5']['s010'][1] - dv('e:transp', 'v2')) < 0.006
PQ = SM['pairs']['in64']; num('SsbLo', rd(PQ['in64_ssb_hard'][0])); num('SsbRef', rd(PQ['in64_ssb_hard'][2])); num('SsbHi', rd(PQ['in64_ssb_hard'][4])); num('InatLo', rd(PQ['in64_inaturalist'][0])); num('InatRef', rd(PQ['in64_inaturalist'][2])); num('InatHi', rd(PQ['in64_inaturalist'][4]))
W['numbers.tex'] = '% written by gen/mk.py\n' + ''.join(f'\\newcommand{{\\n{k}}}{{{v}}}\n' for k, v in NUM.items())
for fn, txt in W.items(): open(os.path.join(OUT, fn), 'w', encoding='utf-8').write(txt)
print('written', len(W), 'files;', len(NUM), 'numbers')
print('pairs where a prior method is above Ours (test):', [(NAME[t], SETN.get(o, o), b) for t, o, b in LOSS])
for k in ('OursT', 'OursLoT', 'OursHiT', 'OursC', 'UdT', 'HfT', 'FprT', 'PriorT', 'GapPriorT', 'GapPriorLoT', 'GapPriorHiT', 'PriorWinT', 'PriorTieT', 'PriorLossT', 'TrainedT', 'GapTrainedT', 'NoUnitT', 'GapNoUnitT', 'SevenT', 'GapSevenT', 'SevenWinT', 'SevenTieT', 'SevenLossT',
          'ValTrOne', 'ValTrTwo', 'ValNoneOne', 'ValNoneTwo', 'ValFullTrOne', 'ValFullTrTwo', 'ValOwnOne', 'ValOwnTwo', 'PassX', 'AllX', 'NLossPairs', 'NLossSvhn', 'NSameU', 'AucTrMin', 'RhoRows', 'SwapRatioMin', 'SwapRatioMax', 'SwapRhoMin', 'SwapRhoMax', 'EqTrHfMin', 'EqTrHfMax'): print(k, NUM[k], end=' | ')
print()
