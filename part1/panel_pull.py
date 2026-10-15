"""Pull one panel dataset from the Census (2025-11-08) with the panel cell filter, thin large non-probe types.
python3 panel_pull.py <dataset_id> <out.h5ad> '<probe1;probe2;...>'"""

import sys, numpy as np, pandas as pd, cellxgene_census

ds, out, PROBE = sys.argv[1], sys.argv[2], sys.argv[3].split(';')
with cellxgene_census.open_soma(census_version='2025-11-08') as c:
    o = (
        c['census_data']['homo_sapiens']
        .obs.read(
            value_filter=f"dataset_id == '{ds}' and is_primary_data == True",
            column_names=[
                'soma_joinid',
                'donor_id',
                'cell_type',
                'assay',
                'tissue_type',
            ],
        )
        .concat()
        .to_pandas()
    )
    for k in ('donor_id', 'cell_type', 'assay', 'tissue_type'):
        o[k] = o[k].astype(str)
    o = o[
        (o.tissue_type == 'tissue')
        & ~o.donor_id.str.contains('pool|unknown', case=False)
        & ~o.cell_type.isin({'unknown', 'cell', 'native cell'})
    ]
    o = o[o.assay == o.assay.value_counts().idxmax()]
    rng = np.random.default_rng(12345)
    vc = o.cell_type.value_counts()
    keep = (vc <= 9500) | vc.index.isin(
        PROBE
    )  # probes never thinned: split stays as panel_census checked it
    ids = [o[o.cell_type.isin(vc.index[keep])].soma_joinid.values]
    for t in vc.index[~keep]:
        ids.append(
            rng.choice(o[o.cell_type == t].soma_joinid.values, 3000, replace=False)
        )
    ids = np.sort(np.concatenate(ids))
    print(ds, len(o), 'filtered cells,', len(ids), 'pulled', flush=True)
    A = cellxgene_census.get_anndata(
        c,
        'Homo sapiens',
        X_name='raw',
        obs_coords=ids,
        obs_column_names=[
            'soma_joinid',
            'donor_id',
            'cell_type',
            'assay',
            'tissue',
            'dataset_id',
        ],
    )
A = A[:, np.asarray(A.X.sum(0)).ravel() > 0].copy()
A.write_h5ad(out, compression='gzip')
print(A.shape, 'written', out, flush=True)
