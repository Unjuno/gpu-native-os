# GPU Native OS

GPU単体での実行を目指すネイティブOSの研究・開発。

Research on an operating system designed around GPU-native execution.

## Status / 開発状況

Research prototypes only. Not a bootable OS, not a GPU timing simulator, and not a security or performance claim.

CPU上での機能検証から始め、正確な状態遷移を基準に、階層型行列命令・動的バッチ・進化的最適化を実験する。最終目標はホストCPUに依存しないGPU主体の実行基盤。必要に応じた最小CPU構成も比較対象とする。

## Principles / 開発原則

- Define exact reference semantics before optimizing.
- Preserve observable state, memory and I/O semantics under optimization.
- Separate CPU functional simulation, GPU simulation and real-device measurements.
- Compare against direct integer execution and simple fusion, not only an intentionally slow baseline.
- Publish reproducible experiments and negative results; do not equate correctness with security.

## License

ライセンスは未決定。公開リポジトリであること自体は再利用許諾を意味しない。
