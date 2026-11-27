# GPU Native OS

GPU単体での実行を目指すネイティブOSの研究・開発。

Research on an operating system designed around GPU-native execution.

## Status / 開発状況

Research prototypes only. Not a bootable OS, not a GPU timing simulator, and not a security or performance claim.

CPU上で機能検証を行い、正確な状態遷移を基準として、階層型行列命令・動的バッチ・進化的最適化を実験する。最終目標はホストCPUに依存しないGPU主体の実行基盤。最小CPU構成も比較対象とする。

## Quick start / 実行

Linux, Python 3.13, NumPy 2.3.5. GCC is required only for the optional native CPU benchmark.

```sh
git clone https://github.com/Unjuno/gpu-native-os.git
cd gpu-native-os
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/run_experiment.py v01
python scripts/run_experiment.py v02 --generations 1 --population 4
# Full search and generated C comparison (Linux/GCC):
python scripts/run_experiment.py v02 --generations 24 --population 12 --native
```

Each experiment copies its source into a new ignored `runs/<run-id>/work/` directory, keeps raw data and logs, and records the environment and source hashes in `manifest.json`. Existing run IDs are rejected. Do not run Python with -O: archived experiments use assertions for checks. Do not execute the archived v02 scripts in place if you want to preserve historical results.

## Repository layout / 構成

| Path | Role |
| --- | --- |
| [experiments/v01](experiments/v01/README.md) | 仮想GPUとゲストOS、分散タスク管理 |
| [experiments/v02](experiments/v02/README.md) | 整数行列命令、参照実装、進化探索、C生成 |
| [tests](tests/test_prototypes.py) | 回帰テスト |
| [scripts](scripts/run_experiment.py) | 実験の分離・再現・記録 |
| [docs/architecture.md](docs/architecture.md) | 目標設計と境界 |
| [docs/experiments.md](docs/experiments.md) | 実験計画・合否条件 |
| [docs/roadmap.md](docs/roadmap.md) | 次の開発課題と完了条件 |
| runs/ | ローカルの生成データ（Git管理外） |

v01 and v02 are separate prototypes, not an integrated OS. Historical results are preserved with their original scripts. See the experiment notes for scope and limitations.

## Development principles

- Define exact reference semantics before optimizing.
- Preserve observable state, memory and I/O semantics.
- Separate CPU functional simulation, GPU simulation and real-device measurements.
- Compare against direct integer execution and simple fusion, not only a slow baseline.
- Publish reproducible experiments and negative results. Correctness is not security.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development loop.

## License

ライセンスは未決定。公開リポジトリであること自体は再利用許諾を意味しない。
