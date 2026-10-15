# Container

Two images, defined in `docker/` and built by `.github/workflows/build-images.yml` on every published release (tagged with the release tag and `latest`) or by manual dispatch. Untested: no image has been built or run yet. They are amd64 only; on Apple silicon add `--platform linux/amd64` (emulated, slow).

| Image | Contents | Use for |
|---|---|---|
| `ghcr.io/fpera0248/recomb2027-hiddenrare` | Python 3.12, `requirements.txt`, this repository at `/opt/recomb2027`, `reproduce` on the PATH | `reproduce paper`, `reproduce rerun`, running our method on a pool, the notebooks |
| `ghcr.io/fpera0248/recomb2027-tools` | R 4.6.1, Seurat 4.4.0, CellSIUS, GapClust, SCISSORS, the Python tools, the official sources at the commits of `tools/sources.txt` under `/opt/tools_src` | `scripts/run_tool.sh` |

No GPU is needed. Image sizes and build times are unknown.

## Run

```
docker pull ghcr.io/fpera0248/recomb2027-hiddenrare:latest
docker run -v "$PWD/out":/out ghcr.io/fpera0248/recomb2027-hiddenrare:latest \
    env ASSETS=/out/assets OUT=/out/reproduce_out reproduce paper
```
To work offline, or for another release tag, put the release files in `out/release` and add `-e ASSET_SRC=/out/release` (or set `TAG`). A missing release gives 404.

`reproduce rerun` and `reproduce full` work the same way (`full` also wants `HR_DATA=/out/work`, disk use not measured; the downloads alone are 29.3 GB). One pool:
```
docker run -v "$PWD/out":/out ghcr.io/fpera0248/recomb2027-hiddenrare:latest \
    scripts/run_pool.sh /out/POOL.h5ad /out/PILOT.parquet /out/run
```
A published tool on an adapted pool (`tools/adapt_pool.py` writes the inputs; the scGPT checkpoint is not in the image):
```
docker run -v "$PWD/out":/out -v /path/to/scGPT_human:/ckpt -e SCGPT_CKPT=/ckpt \
    ghcr.io/fpera0248/recomb2027-tools:latest scripts/run_tool.sh sccad POOL_ID /out/toolin /out/run
```

## Apptainer

```
apptainer pull recomb2027.sif docker://ghcr.io/fpera0248/recomb2027-hiddenrare:latest
apptainer exec -B "$PWD/out":/out recomb2027.sif env ASSETS=/out/assets OUT=/out/reproduce_out reproduce paper
```
`Apptainer.def` builds the same from the published image.

## Build locally

```
docker build -f docker/Dockerfile.hiddenrare -t recomb2027-hiddenrare .
docker build -f docker/Dockerfile.tools -t recomb2027-tools .
```
