"""Census scan for the panel: which datasets have probe cell types with enough donors, eval cells and train cells.  python3 panel_census.py report"""
import os, sys, numpy as np, pandas as pd
VER, NEV, TOP, ORDERS, NEED = '2025-11-08', 200, 128, 20, 18
BAD = {'unknown', 'cell', 'native cell'}
COLS = ['dataset_id', 'donor_id', 'cell_type', 'cell_type_ontology_term_id', 'assay', 'tissue_general', 'tissue_type']
GENERIC = ['cell', 'eukaryotic cell', 'abnormal cell', 'neoplastic cell', 'malignant cell', 'leukocyte', 'lymphocyte',
           'mononuclear leukocyte', 'myeloid cell', 'hematopoietic cell', 'epithelial cell', 'stromal cell',
           'mesenchymal cell', 'glial cell', 'neuron', 'kidney cell', 'neural cell', 'brain vascular cell',
           'supporting cell', 'interstitial cell', 'perivascular cell']
VITRO = 'in vitro|stimulat|activated|cultur|organoid'
MIN_DON = 10
OBO = os.environ['CL_OBO']   # older CL release; labels not in it are never probes

def split_ok(D, types, seed):
    """hlca.py's split on a donor x type count table"""
    M = D[types].to_numpy(); order = np.random.default_rng(seed).permutation(len(M)); ev = np.zeros(len(types))
    for i, d in enumerate(order):
        if (ev >= NEV).all(): break
        ev += M[d]
    else: i = len(order)
    return bool(((M.sum(0) - ev) >= TOP).all() and (ev >= NEV).all())

def rate(D, types): return sum(split_ok(D, types, s) for s in range(ORDERS)) / ORDERS

if sys.argv[1] == 'pull':
    import cellxgene_census
    with cellxgene_census.open_soma(census_version=VER) as c:
        obs = c['census_data']['homo_sapiens'].obs; parts = []
        for k, t in enumerate(obs.read(value_filter='is_primary_data == True', column_names=COLS)):
            parts.append(t.to_pandas().astype(str).groupby(COLS).size())
            if k % 20 == 0: print(k, sum(p.sum() for p in parts), flush=True)
        pd.concat(parts).groupby(level=list(range(len(COLS)))).sum().rename('n').reset_index().to_parquet('panel_counts.parquet')
        c['census_info']['datasets'].read().concat().to_pandas().to_parquet('panel_datasets.parquet')
    print('pulled')
elif sys.argv[1] == 'report':
    name, par = {}, {}; cur = None
    for line in open(OBO):
        line = line.rstrip('\n')
        if line.startswith('['): cur = None
        elif line.startswith('id: CL:'): cur = line[4:]
        elif cur and line.startswith('name: '): name.setdefault(line[6:], cur)
        elif cur and line.startswith('is_a: '): par.setdefault(cur, set()).add(line[6:].split(' ')[0])
    memo = {}
    def anc(t):
        if t not in memo: memo[t] = set().union(*[{q} | anc(q) for q in par.get(t, ())]) if par.get(t) else set()
        return memo[t]
    gen = {name[x] for x in GENERIC} | {'CL:0000003', 'CL:0000548', 'CL:0002371'}   # + obsolete native/animal/somatic cell; KeyError = renamed term
    gen |= {i for n, i in name.items() if 'sensu' in n and not any(v in n for v in ('Vertebrata', 'Mammalia'))}
    C = pd.read_parquet('panel_counts.parquet'); C = C[~C.cell_type.isin(BAD)]
    tid = C.drop_duplicates(['dataset_id', 'cell_type']).set_index(['dataset_id', 'cell_type']).cell_type_ontology_term_id
    meta = pd.read_parquet('panel_datasets.parquet').set_index('dataset_id')
    rows = []
    for ds, g in C.groupby('dataset_id'):
        r = dict(dataset_id=ds, collection_id=meta.collection_id.get(ds), title=meta.dataset_title.get(ds), cells_all=int(g.n.sum()))
        g = g[(g.tissue_type == 'tissue') & ~g.donor_id.str.contains('pool|unknown', case=False)]
        if len(g): g = g[g.assay == g.groupby('assay').n.sum().idxmax()]
        D = g.pivot_table(index='donor_id', columns='cell_type', values='n', aggfunc='sum', fill_value=0) if len(g) else pd.DataFrame()
        r.update(cells=int(g.n.sum()), donors=len(D), types=D.shape[1], assay=g.assay.iloc[0] if len(g) else '',
                 tissue=g.groupby('tissue_general').n.sum().idxmax() if len(g) else '', eligible=0, probe='', qualifies=False)
        if len(D) >= MIN_DON:
            ids = {x: tid[(ds, x)] for x in D.columns}
            iso = lambda x: ids[x] in par and ids[x] not in gen and not any(ids[x] in anc(ids[y]) for y in D.columns if y != x)   # CL leaf, not generic
            tot = D.sum(axis=0); elig = [t for t in tot[tot >= NEV + TOP].index if iso(t) and rate(D, [t]) >= NEED / ORDERS]
            probe = []
            for t in sorted(elig, key=lambda t: tot[t]):
                if rate(D, probe + [t]) >= NEED / ORDERS: probe.append(t)
                if len(probe) == 8: break
            r.update(eligible=len(elig), probe=';'.join(probe), qualifies=len(probe) == 8,
                     probe_min_cells=int(tot[probe].min()) if probe else 0)
        r['donor_ids'] = set(D.index)
        rows.append(r)
    R = pd.DataFrame(rows); R.loc[R.title.str.contains(VITRO, case=False, na=False), 'qualifies'] = False
    Q = R[R.qualifies].sort_values('donors', ascending=False)
    import re
    WORDID = r'(donor|patient|sample|subject|normal|control|healthy|case|ctrl|pat|hc|pt|p|d|s)\d{1,3}'
    def real(ids):
        n = {re.sub(r'[^a-z0-9]', '', d.lower()) for d in ids}
        return {d for d in n if len(d) >= 4 and not d.isdigit() and not re.fullmatch(WORDID, d)}
    unit = {}   # union-find over collection + shared donors (>= 3 non-generic ids)
    def root(x):
        while unit.get(x, x) != x: x = unit[x]
        return x
    q = Q.to_dict('records')
    for a in q: unit[a['dataset_id']] = a['dataset_id']
    pos = {a['dataset_id']: i for i, a in enumerate(q)}
    for i, a in enumerate(q):
        for b in q[:i]:
            if a['collection_id'] == b['collection_id'] or len(real(a['donor_ids']) & real(b['donor_ids'])) >= 3:
                ra, rb = sorted([root(a['dataset_id']), root(b['dataset_id'])], key=pos.get)
                if ra != rb: unit[rb] = ra   # the earlier (more donors) root always wins
    R['unit'] = R.dataset_id.map(lambda x: root(x) if x in unit else '')
    R['panel'] = R.dataset_id.isin([root(x) for x in unit])   # most-donor dataset of each unit (q is donor-sorted, roots come first)
    R = R.drop(columns='donor_ids')
    R.to_csv('panel_eligible.csv', index=False); Pn = R[R.panel]
    print(f'{len(R)} datasets, {R.collection_id.nunique()} studies (human, primary cells, Census {VER})')
    print(f'>= {MIN_DON} clean donors: {(R.donors >= MIN_DON).sum()}; >=8 CL-leaf eligible types: {(R.eligible >= 8).sum()}; '
          f'qualifying: {len(Q)} datasets in {Q.collection_id.nunique()} studies')
    SYN = {'respiratory system': 'lung', 'large intestine': 'colon', 'central nervous system': 'brain'}
    print(f'tissues after merging near-synonyms: {Pn.tissue.replace(SYN).nunique()}')
    print(f'panel (one per donor-independent unit): {len(Pn)} datasets, {Pn.tissue.nunique()} tissues, median {int(Pn.cells.median())} cells, '
          f'median {int(Pn.donors.median())} donors')
    print('panel per tissue:\n' + Pn.tissue.value_counts().to_string())
    print('panel per assay:\n' + Pn.assay.value_counts().to_string())
    for x in ('c838aec3', '218acb0f', '9f222629'):
        m = R[R.dataset_id.str.startswith(x)]; print(x, m[['donors', 'eligible', 'qualifies', 'panel']].to_dict('records'))

if __name__ == '__main__' and sys.argv[1] == 'test':   # the split rule on a toy table
    D = pd.DataFrame({'a': [300, 300, 0], 'b': [100, 100, 100]}, index=['d1', 'd2', 'd3'])
    assert rate(D, ['a']) == 1.0 and rate(D, ['b']) == 0.0; print('test ok')
