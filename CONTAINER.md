# Container

The two images are defined in `docker/`. `.github/workflows/build-images.yml` builds them on every published release (tagged with the release tag and `latest`) or by manual dispatch (tagged with the branch name and `latest`).

Both are published on GHCR with the release tag `v0.1.2` and pull without a login. Use `v0.1.2`. The `v0.1.1` images exist but no check has run them (see [Releases](REPRODUCE.md#releases)), and `latest` and `main` move with later builds.

Test status: `recomb2027-hiddenrare` passes the `paper` and `rerun` levels (`.github/workflows/container-check.yml`, manual). In [this container check](https://github.com/fpera0248/recomb2027/actions/runs/38940072528) on a GitHub Actions runner, `paper` took 3 min 31 s including the image pull and the data download, and `rerun` took 2 min 10 s. The `v0.1.0` image passed the same levels in [an earlier check](https://github.com/fpera0248/recomb2027/actions/runs/38827295374). `recomb2027-tools` has not been run end to end.

The images are amd64 only. On Apple silicon add `--platform linux/amd64` (emulated, slow).

| Image | Contents | Use for |
|---|---|---|
| `ghcr.io/fpera0248/recomb2027-hiddenrare` | Python 3.12, `requirements.txt`, this repository at `/opt/recomb2027`, `reproduce` on the PATH | `reproduce paper`, `reproduce rerun`, running our method on a pool, the notebooks |
| `ghcr.io/fpera0248/recomb2027-tools` | R 4.6.1, Seurat 4.4.0, CellSIUS, GapClust, SCISSORS, the Python tools, the official sources at the commits of `tools/sources.txt` under `/opt/tools_src` | `scripts/run_tool.sh` |

No GPU is needed.

## Run

```
docker pull ghcr.io/fpera0248/recomb2027-hiddenrare:v0.1.2
docker run -v "$PWD/out":/out ghcr.io/fpera0248/recomb2027-hiddenrare:v0.1.2 \
    env ASSETS=/out/assets OUT=/out/reproduce_out reproduce paper
```
For offline use or another release tag, put the release files in `out/release` and add `-e ASSET_SRC=/out/release` (or set `TAG`). A missing release gives 404.

`reproduce rerun` and `reproduce full` work the same way. For `full`, also pass `HR_DATA=/out/work`, because the default, ./work, is inside the container. `full` downloads 29.3 GB. Its total disk use is not measured.

To run one pool:
```
docker run -v "$PWD/out":/out ghcr.io/fpera0248/recomb2027-hiddenrare:v0.1.2 \
    scripts/run_pool.sh /out/POOL.h5ad /out/PILOTS.parquet /out/run
```
A published tool on an adapted pool (`tools/adapt_pool.py` writes the inputs, and the scGPT checkpoint is not in the image):
```
docker run -v "$PWD/out":/out -v /path/to/scGPT_human:/ckpt -e SCGPT_CKPT=/ckpt \
    ghcr.io/fpera0248/recomb2027-tools:v0.1.2 scripts/run_tool.sh sccad POOL_ID /out/toolin /out/run
```

## Apptainer

```
apptainer pull recomb2027.sif docker://ghcr.io/fpera0248/recomb2027-hiddenrare:v0.1.2
mkdir -p out
apptainer exec -B "$PWD/out":/out recomb2027.sif env ASSETS=/out/assets OUT=/out/reproduce_out reproduce paper
```
`Apptainer.def` builds the same from the published image.

Digests of the `v0.1.2` images:

- recomb2027-hiddenrare sha256:55e2273ae7f7689c86bf35ea83ed2b83db191c3ae0548dd1ed55f377ae8a4e46
- recomb2027-tools sha256:190f2d8e306cb1ef884c125467e90a8d86a3ae40aba08f3bfe35b6e054ff416b

Digests of the `v0.1.0` images:

- recomb2027-hiddenrare sha256:6d856f7fc7c5adcf722ef89b83bbd7671edc5e29e454f08b9524b96f059de283
- recomb2027-tools sha256:07ab7720f7ff1f6ff65f7b1f765a46db81b4d7e4125680d21a42d44f901811b6

## Build locally

```
docker build -f docker/Dockerfile.hiddenrare -t recomb2027-hiddenrare .
docker build -f docker/Dockerfile.tools -t recomb2027-tools .
```
