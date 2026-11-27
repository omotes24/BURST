<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/burst-logo-dark.svg">
    <img src="assets/burst-logo.svg" alt="BURST" width="340">
  </picture>
</p>

<h3 align="center">破れた対称性による拡散モデルの分布外検知</h3>

<p align="center">
  表 紘太朗&emsp;尾崎 慶次郎&emsp;中澤 仁<br>
  <sub>慶應義塾大学</sub>
</p>

<p align="center">
  <a href="paper/BURST_JA_1127c.pdf"><b>原稿 PDF</b></a>&emsp;·&emsp;<a href="code/burst_score.py"><b>コード</b></a>&emsp;·&emsp;<a href="#引用"><b>引用</b></a>
</p>

<table align="center">
  <tr>
    <td align="center"><b>96.5%</b><br><sub>平均 AUROC</sub></td>
    <td align="center"><b>13.8%</b><br><sub>FPR95</sub></td>
    <td align="center"><b>1回</b><br><sub>途中で止める順伝播</sub></td>
    <td align="center"><b>0</b><br><sub>学習するパラメータ</sub></td>
  </tr>
</table>

<p align="center">
  <img src="assets/fig1_teaser.png" width="100%" alt="転置1つが距離の単位になる">
</p>

## BURST とは

入力から train 画像までの最近傍の距離には、基準になる長さがありません。BURST は、train 画像とその転置(縦と横の入れ替え)をノイズ除去器に通したときの記述子の距離 $`\tau`$ を「長さの単位」にして、入力までの距離 $`r`$ をそれで割ります。自然画像は縦と横を区別するので、転置した train 画像は「内容は分布内と同じで、分布内ではない画像」になります。

```math
s(x)=\sum_{g\in\lbrace f,z\rbrace}d_g\sum_{k=1}^{N}\frac{1/k}{H_N}\log\frac{\tau_g(i_k)}{r_g(x,i_k)}
```

$`\tau_g(i)=|g(x_i)-g(\mathbf{T}x_i)|`$ は train 画像 $`x_i`$ の転置距離、$`r_g(x,i)=|g(x)-g(x_i)|`$ は入力までの距離、$`i_k`$ は $`k`$ 番目に近い train 画像、$`H_N=1+\frac{1}{2}+\cdots+\frac{1}{N}`$、$`d_g`$ は記述子 $`g`$ の次元です。大きいほど分布内らしいスコアです。

<p align="center">
  <img src="assets/fig2_method.png" width="100%" alt="BURST の流れ">
</p>

- **2つの記述子**　$`f`$ は読み出しブロック U の出力のチャネルごとの平均と標準偏差、$`z`$ は打ち切りブロック X までの各ブロックの振幅です。
- **事前**　train 画像をそのままと転置の2通りで通し、1枚につき1つの数 $`\tau`$ だけを持ちます。
- **推論**　X で止める1回の順伝播と、距離の重みつき平均だけです。近傍数も選びません。

## 結果

| 手法 | CIFAR‑10 | SVHN | CelebA | ImageNet‑64 | FashionMNIST | 平均 | FPR95↓ |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| DiffPath | 72.4 | **99.8** | 99.8 | 55.5 | 80.6 | 81.6 | 47.4 |
| CFS‑kNN | 85.4 | 93.8 | 99.6 | 61.3 | **99.8** | 88.0 | 35.9 |
| 距離のみ | 92.8 | 97.6 | **100.0** | 78.0 | 99.7 | 93.6 | 24.1 |
| **BURST** | **96.0** | 98.9 | **100.0** | **87.7** | 99.7 | **96.5** | **13.8** |

<sub>test 画像(各集合500枚)での、自然画像の分布外集合の平均 AUROC(%)。CFS‑kNN は同じ拡散モデルを用いる先行研究の最良(CFS の kNN ヘッド)、距離のみは BURST から転置距離を除いたものです。BURST と CFS‑kNN はノイズの種3つの平均です。10の先行手法すべてとの比較は原稿の表1にあります。転置を単位にする選択は、test 画像での値も見たうえで決めました(原稿5.1節)。</sub>

## 使い方

```python
from burst_score import build_bank, burst   # code/burst_score.py

bank_f = build_bank(Gf, unit=True)   # Gf: [N, 2, D]  train 画像の f(そのまま、転置)
bank_z = build_bank(Gz)              # Gz: [N, 2, X]  train 画像の z(そのまま、転置)
s = burst(Qf, Qz, bank_f, bank_z)    # 大きいほど分布内らしい
```

$`f`$ は長さ1に正規化し、$`z`$ は正立の train 画像の平均と標準偏差で標準化してから渡します。記述子は、凍結したノイズ除去器を打ち切りブロックまで1回通して求めます(この部分はリポジトリに含みません)。

## 構成

```
paper/BURST_JA_1127c.pdf   原稿(日本語、本文と参考文献の9ページ)
code/burst_score.py        スコアの参照実装
assets/                    ロゴと図
```

## 引用

```bibtex
@misc{omote2026burst,
  title  = {{BURST}：破れた対称性による拡散モデルの分布外検知},
  author = {表, 紘太朗 and 尾崎, 慶次郎 and 中澤, 仁},
  year   = {2026},
  url    = {https://github.com/omotes24/BURST}
}
```

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/burst-mark-dark.svg">
    <img src="assets/burst-mark.svg" alt="" width="28">
  </picture>
</p>
