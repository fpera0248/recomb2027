"""Unified figure style for the RECOMB 2027 paper (two-column ACM, 3.08 in column, 6.50 in full text width).
Usage:  from paperstyle import apply_style, color, label, panel_label, legend_below, save, COL1, COL2
Rules:
 * One colour per method, identical in every figure. Hue = method family, lightness = variant
   (as published = family colour, label-aware variant = LIGHT version, same hue).
 * Our method is the ONLY maroon. Leiden subclustering is always mid grey. Wrong calls are always black.
 * Hidden (target) cells are always TEAL. Visible broad labels use the categorical LABEL_CYCLE, never a method colour.
 * No titles inside images, no dashed frame, no codenames. Panel letters (a), (b) bold at top-left.
 * Font: Linux Biolinum if installed (matches ACM sans), else Helvetica/Arial, else DejaVu Sans. 9 pt base, 8 pt min, 10 pt max.
"""
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

COL1, COL2 = 3.08, 6.50     # inches, measured by compiling main.tex (acmart sigconf, margin=1in): columnwidth 222.88 pt, textwidth 469.755 pt. Draw at inclusion size.
FS, FS_MIN, FS_MAX = 9.0, 8.0, 10.0   # 9 pt = existing paper convention (make_paper_figs.py); RECOMB asks min 10 pt, so never go below 8
LW, LW_THIN, LW_BAR = 1.0, 0.6, 0.5      # lines, axes/ticks/error bars, bar edges
MS, MS_UMAP = 4.5, 1.0                   # marker size (pt) for dots, UMAP point size (pt^2, rasterised)

# ---- Part II (hidden-type discovery) ----------------------------------------------------------
OURS = '#7C2529'
COLORS = {
  'ours': OURS, 'ours_nolabel': '#B5585C',                      # our method with / without visible labels
  'leiden': '#6E6E6E', 'leiden_nolabel': '#A8A8A8',             # Leiden subclustering
  'sccad': '#0072B2', 'sccad_lab': '#7FB4D8',                   # scCAD as published / label-aware (our modification)
  'giniclust3': '#E69F00', 'giniclust3_lab': '#F3CC80', 'giniclust3_nf': '#FFD966',   # _nf: no filters, 5 neighbors (Fig 3 only)
  'fire': '#2E8B57', 'fire_lab': '#8FC9A8',
  'annogcd': '#CC79A7',
  'cellsius_label': '#5E8C9A', 'gapclust': '#C23B7A', 'scgpt_if': '#9A8A2E',   # extra tools of Fig 3a
  'iforest_pca': '#7A6A1E', 'iforest_scvi': '#B5A24A', 'lof_pca': '#D6CB8E',   # anomaly scores: olive family
  'uncertainty': '#4B3B7A', 'dup_loop': '#7C6BB0', 'nodup_loop': '#A89CD0', 'classweight_loop': '#DCC3EA',  # pseudo-label loops: violet family
  'random': '#000000',                                           # random ranking: black dashed line, never a bar
}
# ---- Part I (augmentation vs generators) -------------------------------------------------------
COLORS.update({
  'duplication': '#1F3A5F',                                      # reference method of Part I (navy)
  'noaug': '#C8C8C8',
  'noise': '#78A31A', 'noise_nz': '#C5E07A', 'smote': '#A8329A', 'copy_spread': '#6FA8A0', 'copy_encdec': '#9CCBC4',
  'sd3_published': '#D95F02', 'sd3_nb': '#F4A261',               # scDesign3 as published / copula deleted (negative-binomial sampler)
  'scvi': '#3B3B1E', 'scvi_cond': '#8D6E63', 'scdiffusion': '#A5662B', 'cfgen_shipped': '#D9A38C', 'cfgen_best': '#8A8A55',  # trained generators: brown family (no red, far from our maroon)
})
# ---- roles ---------------------------------------------------------------------------------------
HIDDEN = '#00A6A6'     # hidden (target) cells in every UMAP and in every legend
WRONG = '#000000'      # a wrong call is always black
CHANCE = '#000000'     # dashed line
LABEL_CYCLE = ['#62D957', '#AD57D9', '#A3D96C', '#57D9B8', '#5757D9', '#47B274', '#578DD9', '#C9A227']  # visible labels: no grey, no method hue (checked by the self-check); cycle, 'other' = BACKGROUND_CELLS
BACKGROUND_CELLS = '#E3E3E3'
# subtypes drawn in more than one figure keep one colour (Fig 3b and Supplementary Fig S1b); the values are the colours Fig 3b assigns
SUBTYPE_COLORS = {'T.CD8.Predysfunc': '#62D957', 'T.CD8.Cytotoxic': '#AD57D9', 'T.CD8.Exhausted': '#C9A227', 'T.CD8.Naive': '#5757D9',
                  'stromal_VSTM2A+': '#62D957', 'stromal_1': '#AD57D9', 'stromal_2': '#C9A227', 'stromal_transitional_3': '#5757D9',
                  'CLP': '#62D957', 'MEBEMP-L': '#AD57D9', 'ERYP': '#C9A227', 'BEMP': '#5757D9', 'NKTDP': '#578DD9'}
DIVERGING = 'RdBu'     # heatmaps of differences: centre 0, blue = better than reference, red = worse; always state the reference

LABELS = {  # display names, the only strings allowed in axes and legends
  'ours': 'Our method', 'ours_nolabel': 'Our method (no labels)', 'leiden': 'Leiden subclustering', 'leiden_nolabel': 'Leiden subclustering (no labels)',
  'sccad': 'scCAD (default)', 'sccad_lab': 'scCAD (per label)', 'giniclust3': 'GiniClust3 (default, swapped filters)', 'giniclust3_lab': 'GiniClust3 (per label, swapped filters)',
  'fire': 'FiRE (default)', 'fire_lab': 'FiRE (per label)', 'annogcd': 'AnnoGCD',
  'iforest_pca': 'Isolation forest, PCA', 'iforest_scvi': 'Isolation forest, scVI', 'lof_pca': 'Local outlier factor, PCA', 'uncertainty': 'Classifier uncertainty',
  'dup_loop': 'Self-training loop, duplication', 'nodup_loop': 'Self-training loop, no duplication', 'classweight_loop': 'Self-training loop, class weights', 'random': 'Random ranking',
  'duplication': 'Duplication', 'noaug': 'No augmentation', 'noise': 'Noise on copies', 'noise_nz': 'Noise, nonzero genes only', 'giniclust3_nf': 'GiniClust3 (no filters, 5 neighbors)', 'cellsius_label': 'CellSIUS (per label)', 'gapclust': 'GapClust (earlier run)', 'scgpt_if': 'scGPT with isolation forest', 'smote': 'SMOTE', 'copy_spread': 'Real latent codes, spread out',
  'copy_encdec': 'Real latent codes, decoded', 'sd3_published': 'scDesign3 (default)', 'sd3_nb': 'scDesign3 without copula',
  'scvi': 'scVI', 'scvi_cond': 'Conditional scVI', 'scdiffusion': 'scDiffusion', 'cfgen_shipped': 'CFGen (default)', 'cfgen_best': 'CFGen, class-balanced minibatches',
}
CLASSIFIER_MARKERS = {'logistic regression': 'o', 'k-nearest neighbours': '^', 'random forest': 's', 'multilayer perceptron': 'D'}  # Part I marker = classifier, always

# ---- every figure must contain only keys of COLORS; unknown key -> KeyError (no hash-based fallbacks)
def color(key): return COLORS[key]
def label(key): return LABELS[key]
def lighten(c, f=0.5): r = to_rgb(c); return tuple(x + (1 - x) * f for x in r)

def apply_style():
    for fam in ('Arial', 'DejaVu Sans'):   # Arial: the paper figures use one family, and it has a registered bold face for panel letters
        from matplotlib import font_manager as fm
        if any(f.name == fam for f in fm.fontManager.ttflist): break
    mpl.rcParams.update({
      'font.family': 'sans-serif', 'font.sans-serif': [fam, 'DejaVu Sans'], 'font.size': FS,
      'axes.titlesize': FS, 'axes.labelsize': FS, 'xtick.labelsize': FS_MIN, 'ytick.labelsize': FS_MIN, 'legend.fontsize': FS_MIN,
      'axes.titleweight': 'normal', 'axes.linewidth': LW_THIN, 'xtick.major.width': LW_THIN, 'ytick.major.width': LW_THIN,
      'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'lines.linewidth': LW, 'lines.markersize': MS, 'patch.linewidth': LW_BAR,
      'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False, 'axes.axisbelow': True,
      'legend.frameon': False, 'legend.handlelength': 1.4, 'legend.handletextpad': .5, 'legend.columnspacing': 1.2, 'legend.borderaxespad': 0,
      'figure.dpi': 150, 'savefig.dpi': 600, 'savefig.facecolor': 'white', 'figure.facecolor': 'white',
      'pdf.fonttype': 42, 'ps.fonttype': 42, 'figure.constrained_layout.use': True, 'image.cmap': 'viridis',
      'errorbar.capsize': 1.5,
    })

def panel_label(ax, letter, dx=-0.02, dy=1.04):
    """bold '(a)' above the top-left corner of an axes; use on every multi-panel figure, none on single-panel ones."""
    ax.text(dx, dy, f'({letter})', transform=ax.transAxes, fontweight='bold', fontsize=FS + .5, ha='right', va='bottom')

KEY_LABELS = {'uses_labels': 'Uses broad labels', 'no_labels': 'No labels', 'atlas_mean': 'Atlas mean (open circle)', 'random': 'Random ranking'}   # key text shared by Figures 1d and 2a

def panel_letter(fig, letter, x_in, y_in):
    """The one panel-letter rule for every paper figure: bold '(a)' in FS+0.5 pt, left edge at x_in and baseline bottom at y_in (inches from the figure's lower-left corner), at the top-left corner of its panel block."""
    w, h = fig.get_size_inches(); fig.text(x_in / w, y_in / h, f'({letter})', fontweight='bold', fontsize=FS + .5, ha='left', va='bottom')

def legend_below(fig_or_ax, handles=None, labels=None, ncol=3, **kw):
    """Legend rules: (1) ONE figure-level legend below the panels covers every panel, say so in the caption; (2) a legend
    that applies to one panel only goes inside that panel and is introduced in the caption as 'legend in (a) applies to (a)'; (3) never over data."""
    f = fig_or_ax.figure if hasattr(fig_or_ax, 'figure') else fig_or_ax
    if handles is None: return f.legend(loc='outside lower center', ncol=ncol, **kw)
    return f.legend(handles, labels, loc='outside lower center', ncol=ncol, **kw)

def save(fig, path_no_ext):
    for e in ('pdf', 'png'): fig.savefig(f'{path_no_ext}.{e}')

if __name__ == '__main__':   # self-check: palette has no near-duplicates; writes swatch.png
    import itertools, numpy as np
    apply_style()
    allc = {**COLORS, **{f'label{i}': c for i, c in enumerate(LABEL_CYCLE)}, 'hidden': HIDDEN, 'background': BACKGROUND_CELLS}
    ks = list(COLORS); bad = []
    dist = lambda a, b: float(np.linalg.norm(np.array(to_rgb(allc[a])) - np.array(to_rgb(allc[b]))))
    for a, b in itertools.combinations(allc, 2):
        if dist(a, b) < 0.12: bad.append((a, b, round(dist(a, b), 3)))
    assert set(LABELS) >= set(ks) - {'random'}, set(ks) - set(LABELS)
    assert sum(c == OURS for c in COLORS.values()) == 1
    near_ours = [(k, round(dist('ours', k), 3)) for k in allc if k != 'ours' and dist('ours', k) < 0.25]
    print('near-duplicate pairs over colours + labels + roles (dist<0.12):', bad)
    print('keys closer than 0.25 to our colour:', near_ours)
    assert not bad and not near_ours
    fig, ax = plt.subplots(figsize=(COL2, 3.2))
    for i, k in enumerate(ks): ax.barh(len(ks) - i, 1, color=COLORS[k], edgecolor='black'); ax.text(1.02, len(ks) - i, f'{k}: {LABELS.get(k, "")}', va='center')
    ax.set_xlim(0, 3); ax.axis('off'); fig.savefig(__file__.replace('paperstyle.py', 'swatch.png'))
