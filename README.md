# ModelBridge Library

**モデルブリッジライブラリ及びサンプルアプリケーション**

ハイパーパラメータ最適化において、計算コストの高いミクロモデルの代わりに高速なマクロモデルを使用し、回帰モデルによってパラメータを推定する技術を提供します。

## 📚 モデルブリッジとは

モデルブリッジは、以下の問題を解決する技術です：
- **ミクロモデル**: 高精度だが計算時間が長い
- **マクロモデル**: 高速だが精度が低い

この2つのモデルを「ブリッジ」することで、**高速でありながら高精度**なハイパーパラメータ最適化を実現します。

### 🔄 モデルブリッジのワークフロー

1. **データ分割**
   - データセットをトレーニング用とテスト用に分割

2. **トレーニング段階**
   - n_train個のデータセットでミクロモデルのハイパーパラメータ最適化
   - 同じデータセットでマクロモデルのハイパーパラメータ最適化
   - マクロ→ミクロパラメータの回帰モデルを学習

3. **テスト段階**
   - n_test個のデータセットでミクロモデル最適化（正解データ）
   - マクロモデル最適化後、回帰モデルでミクロパラメータを予測
   - 予測精度を評価・検証

## 🎯 サンプル実装

### 1. Simple Benchmark
```bash
# 数学関数（sphere, rastrigin, griewank）を使用
# 回帰モデル: 線形・多項式・ガウス過程回帰
cd example/simple_benchmark
python hpopt_benchmark_refactored.py -c config_sample.toml
```

### 2. Neural Network (MNIST)
```bash
# ニューラルネットワークのモデルブリッジ
# 2層MLP（ミクロ）→ 1層MLP（マクロ）
cd example/neural_network
python mnist_sklearn_bridge.py
```

### 3. MAS-Bench
```bash
# 交通シミュレーションのモデルブリッジ
# データ同化によるマクロモデル構築
cd example/mas_bench
python hpopt_data_assimilation_refactored.py
```

詳細: [MAS-Bench](https://github.com/MAS-Bench/MAS-Bench)

## 🚀 クイックスタート

### 基本的な使用方法

```python
from modelbridge import ModelBridge

# 目的関数を定義
def micro_objective(params):
    # 高精度だが時間のかかるモデル
    return expensive_evaluation(params)

def macro_objective(params, target):
    # 高速だが簡素なモデル
    return fast_approximation(params)

# パラメータ設定
param_config = {
    "x1": {"type": "float", "low": -5.0, "high": 5.0},
    "x2": {"type": "float", "low": -5.0, "high": 5.0}
}

# ModelBridge実行
bridge = ModelBridge(
    micro_objective=micro_objective,
    macro_objective=macro_objective,
    micro_param_config=param_config,
    macro_param_config=param_config,
    regression_type="polynomial"
)

# 完全なパイプライン実行
results = bridge.run_full_pipeline(
    n_train=10, n_test=5,
    visualize=True, output_dir="results"
)
```

---

## 🛠️ 開発環境のセットアップ

**モダンなPython開発環境**: `Python 3.12+` + `uv` + `ruff` + `mypy`

### 前提条件
- **Python 3.12以上** が必要です

### インストール

```bash
# 基本インストール
uv pip install -e .

# 開発用依存関係を含む
uv pip install -e ".[dev]"

# すべての依存関係を含む
uv pip install -e ".[all]"
```

### 開発コマンド

| コマンド | 機能 |
|----------|------|
| `make lint` | **ruff**によるコードリント |
| `make format` | **ruff**による自動フォーマット |
| `make type-check` | **mypy**による型チェック |
| `make test` | **pytest**によるテスト実行 |
| `make check-all` | 🔥 **全チェック実行** |

```bash
# 開発環境の品質チェック
make check-all

# カバレッジ付きテスト
make test-cov

# パッケージビルド
make build
```

---

## 📁 プロジェクト構成

```
📦 ModelBridge Library
├── 🧠 modelbridge/           # コアライブラリ
│   ├── core/                 # 主要機能
│   │   ├── optimizer.py      # Optuna最適化エンジン
│   │   ├── regression.py     # ML回帰モデル
│   │   └── bridge.py         # モデルブリッジ調整
│   └── utils/                # ユーティリティ
│       ├── config_loader.py  # 設定管理
│       ├── data_manager.py   # データ処理
│       └── visualization.py  # 結果可視化
├── 📊 example/               # 実装例
│   ├── simple_benchmark/     # 数学関数最適化
│   ├── neural_network/       # 🧠 ニューラルネットワーク
│   └── mas_bench/           # 交通シミュレーション
└── 📋 pyproject.toml        # プロジェクト設定
```

### サポート技術

- **Python**: 3.12+ (最新言語機能活用)
- **最適化**: Optuna (TPE, CMA-ES, Random)
- **回帰**: 線形・多項式・ガウス過程
- **可視化**: Matplotlib自動プロット生成
- **設定**: TOML設定ファイル
- **型安全性**: 完全な型ヒント対応
