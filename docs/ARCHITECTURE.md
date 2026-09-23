# アーキテクチャ設計書 & 技術選定書（Architecture & Design Spec）

本ドキュメントは、Kubeflow Pipelines（KFP SDK v2）および Google Cloud Vertex Pipelines 上で動作するペンギン分類パイプラインのアーキテクチャ、技術選定、コンポーネント設計パターン、およびコード構成仕様を定義する設計書です。

---

## 1. プロジェクト目的と2つの提供価値

本サンプルリポジトリは、Kubeflow Pipelines SDK v2 および Vertex AI Pipelines を用いた実践的な機械学習パイプラインの構築において、開発者が直面する典型的な課題を解決するための実装リファレンスです。用途や既存資産の成熟度に応じて選択できるよう、以下の2つの明確な目的を同時に達成することを目指します。

### 1.1 汎用性と応用性を重視したコンテナパイプライン
既存の Dockerfile 資産や任意の CLI ツールを最小限の手間でパイプライン化する標準パターンを提供します。各コンポーネントに CLI エントリポイントを用意し、コンテナを外部から呼び出すパイプライン（`pipeline_container.py`）を採用します。これにより、言語やフレームワークを問わず、コンテナ化された任意の処理を Vertex Pipelines 上で迅速に結合・実行する手法を習得できます。

### 1.2 KFP SDK Artifacts と Vertex AI 可視化をフル活用したパイプライン
`Input[Dataset]`, `Output[Model]`, `Output[Metrics]`, `Output[ClassificationMetrics]` などの Python オブジェクトやメタデータを直接操作し、Vertex AI の可視化機能を最大限に引き出すモダンなパターンを提供します。コンポーネントパッケージのファサードから公開された Component 関数を直接連携させるパイプライン（`pipeline_component.py`）を採用し、リネージ追跡や混同行列の対話的表示など、最新プラットフォームの恩恵を享受する設計を提示します。

### 1.3 参照公式ドキュメント
- Vertex AI Pipelines 概要: [Vertex AI Pipelines Overview](https://cloud.google.com/vertex-ai/docs/pipelines/overview)
- Kubeflow Pipelines v2 移行ガイド: [KFP v2 Migration Guide](https://www.kubeflow.org/docs/components/pipelines/v2/migration/)

---

## 2. 技術スタック選定と根拠

本章では、本パイプラインの長期的なメンテナンス性、コンテナビルドおよびテストの高速化、そして Vertex AI Pipelines 上での堅牢な可視化・追跡性を実現するために選定した主要技術スタックと、その選定理由を述べます。

### 2.1 Python バージョン: Python 3.14

本プロジェクトではランタイム言語として Python 3.14 を採用しています。公式のサポートステータス、コンテナ運用、主要 ML ライブラリとの適合性の観点からその選定根拠を以下に説明します。

#### 公式ライフサイクル（devguide）に基づく位置づけ
Python 公式開発ガイド（[Status of Python versions](https://devguide.python.org/versions/)）において、各バージョンのフェーズは主に bugfix、security、end-of-life に分類されています。

バグ修正およびセキュリティ修正が定期的にリリースされる公式の主軸フェーズである bugfix（Active Maintenance / Stable）には、現在 Python 3.14（2025年10月初回リリース、EOL: 2030年10月）および Python 3.13（EOL: 2029年10月）が位置しています。これに対し、リリース後一定期間を経てバグ修正が終了し重大なセキュリティ脆弱性のみがソース提供される security フェーズには Python 3.12, 3.11, 3.10 が属しており、Python 3.9 以前はサポートが完全に終了した end-of-life フェーズにあります。

したがって、新しく構築するプロジェクトや書籍・教材等で長期的にメンテナンス性を担保するためには、既に security フェーズに入った 3.11 や 3.12 ではなく、現在 active maintenance（bugfix）フェーズにある Python 3.14 を選定するのが公式ライフサイクルの原則に合致しています。

#### カスタムコンテナによる完全な独立性
本パイプラインは Vertex AI の事前ビルドコンテナ（Prebuilt containers）に制約される構成ではなく、各コンポーネントが独自の `Dockerfile`（`python:3.14-slim-bookworm`）を持つ「カスタムコンテナコンポーネント」です。Google Cloud 側の事前ビルド更新スケジュールに縛られることなく、最新かつ最適なランタイムを自律的に利用できます。

#### 主要 ML ライブラリの対応検証
本プロジェクトで使用する全依存パッケージ（`kfp==2.17.0`, `google-cloud-aiplatform==2.1.3`, `scikit-learn==1.9.1`, `pandas==3.0.6`, `numpy==2.5.3` 等）について、`uv pip compile -p 3.14` にて依存解決を実施し、Python 3.14 向け wheel の正常な解決と互換性を物理的に検証済みです。

#### 参照公式ドキュメント
- Python 公式バージョンライフサイクル: [Status of Python versions](https://devguide.python.org/versions/)
- PEP 745（Python 3.14 リリーススケジュール）: [PEP 745 – Python 3.14 Release Schedule](https://peps.python.org/pep-0745/)

### 2.2 パッケージ・仮想環境管理: uv

#### 選定理由
Astral 社が開発する Rust 製の高速パッケージ管理ツールです。PEP 621 標準（`[project]` 形式）に準拠し、`uv lock` による厳密な再現性を担保します。特にコンテナビルド時に `uv pip install --system` を用いることで、従来の Poetry（`curl | python` 経由の低速インストールや依存解決）と比較してビルド時間を数分から数秒へと大幅に短縮できます。

#### 参照公式ドキュメント
- uv 公式ドキュメント: [Astral uv Documentation](https://docs.astral.sh/uv/)
- uv によるプロジェクト・依存管理: [uv Projects Guide](https://docs.astral.sh/uv/guides/projects/)
- uv を活用した Docker コンテナビルド: [Using uv in Docker](https://docs.astral.sh/uv/guides/docker/)
- Python 標準プロジェクト仕様: [PEP 621 – Storing project metadata in pyproject.toml](https://peps.python.org/pep-0621/)

### 2.3 パイプラインオーケストレーション: Kubeflow Pipelines SDK v2（kfp >= 2.8.0）

#### 選定理由
Vertex Pipelines およびオープンソース Kubeflow Pipelines 双方で統一された中間表現（Intermediate Representation: IR YAML）仕様に準拠しており、Artifact（Dataset, Model, Metrics）の一級市民としての追跡や、パイプラインの再現性・可視化を完全にサポートします。

#### 参照公式ドキュメント
- Kubeflow Pipelines v2 ドキュメント: [Kubeflow Pipelines v2 Documentation](https://www.kubeflow.org/docs/components/pipelines/v2/)
- Vertex AI パイプライン構築ガイド: [Build a pipeline using KFP SDK](https://cloud.google.com/vertex-ai/docs/pipelines/build-pipeline)

### 2.4 コンテナレジストリ: Google Cloud Artifact Registry

#### 選定理由
従来の Container Registry（`*.gcr.io`）は Google Cloud により非推奨化（Deprecated）されており、最新のプロジェクトでは Artifact Registry（`{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPOSITORY}/{IMAGE}:{TAG}`）が唯一の標準レジストリです。

#### 参照公式ドキュメント
- Container Registry からの移行: [Transition from Container Registry to Artifact Registry](https://cloud.google.com/artifact-registry/docs/transition/transition-from-gcr)
- Artifact Registry Docker クイックスタート: [Artifact Registry Docker Quickstart](https://cloud.google.com/artifact-registry/docs/docker/quickstart)

### 2.5 テストフレームワーク: pytest

#### 選定理由
Python エコシステムにおけるデファクトスタンダードのテストフレームワークです。
`tmp_path` フィクスチャを活用したファイル入出力（CSV/Pickle/YAML）の安全な隔離テスト、CLI 引数や終了コードの検証、および KFP SDK v2 のコンポーネント Spec 定義とローカル実行機能（`comp.python_func`）による高速テストを実現します。
ルートの `pyproject.toml`（`[tool.pytest.ini_options]`）の設定により、`uv run pytest` コマンド 1 回で各コンポーネントおよびパイプライン全体の全テストを数秒で一括自動実行できます。

#### 参照公式ドキュメント
- pytest 公式ドキュメント: [pytest: helps you write better programs](https://docs.pytest.org/)

---

## 3. コンポーネント設計（コンテナ・3層分離・ビルド）

Kubeflow Pipelines（KFP）におけるコンポーネントの本質は、「永続ストレージ（Cloud Storage 等）からデータを受け取り、隔離されたコンテナ内で処理を実行し、結果を再び永続ストレージへ書き出す独立したコンテナ」です。
したがって、コンポーネントの設計においては、内部ロジックのモジュール構造だけでなく、コンテナイメージのビルド構成やエントリポイント設計、そして KFP オーケストレータとのインターフェース制約までを一貫した体系として捉える必要があります。

### 3.1 コンポーネントの 3 ファイル構成（Core / CLI / Component）

各コンポーネント（`components/data_generator`, `components/transform`, `components/trainer`, `components/evaluator`）は、関心事の分離（Separation of Concerns）に基づき、以下の命名規則による 3 ファイル構成をとります。純粋なアルゴリズム、CLI インターフェース、およびパイプライン連携アダプターの責務を物理的に分離することで、ローカル単体テストの容易性とパイプラインへの組み込みやすさを両立します。

```text
components/{name}/
├── Dockerfile               # コンポーネントコンテナのビルド定義
├── pyproject.toml           # コンポーネント固有の依存定義（uv 管理）
├── __init__.py              # パッケージファサード: 外部向けに {name}_op を公開
├── src/
│   ├── {name}.py            # 純粋ロジック: KFP/CLI 非依存、トップレベル import、pytest テスト対象
│   ├── {name}_cli.py        # CLI アダプター: argparse 経由で {name}.py を実行（路線1用）
│   └── {name}_component.py  # KFP アダプター: KFP SDK Artifacts 経由で {name}.py を実行（路線2用）
└── tests/
    ├── test_{name}.py           # 純粋ロジックの高速な pytest 単体テスト
    ├── test_{name}_cli.py       # CLI 引数・終了コードの単体テスト
    └── test_{name}_component.py # KFP Component アダプター（入出力型・メタデータ付与）の単体テスト
```

#### ファイル別の責務と実装例（trainer を例として）

ここではモデル学習を担う trainer コンポーネントを題材とし、純粋ロジック、CLI アダプター、KFP Component アダプター、およびテストコードの各ファイルがどのような責務を担い、どのように連携するかを具体的に示します。

##### 純粋ロジック（components/trainer/src/trainer.py）
KFP や argparse に一切依存しない純粋な Python 関数です。トップレベルで通常通り import し、ファイルパスやパラメータを受け取って処理を実行します。

```python
# src/trainer.py
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_model(
    train_data_path: str,
    model_output_path: str,
    suffix: str = "_xf",
    n_estimators: int = 100,
    random_state: int = 42,
) -> dict:
    df = pd.read_csv(train_data_path)
    X = df.drop(columns=[f"species{suffix}"])
    y = df[f"species{suffix}"]

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X, y)

    with open(model_output_path, "wb") as f:
        pickle.dump(clf, f)

    return {
        "framework": "scikit-learn",
        "algorithm": "RandomForestClassifier",
        "n_samples": len(df),
    }
```

##### CLI アダプター（components/trainer/src/trainer_cli.py）
路線1（汎用コンテナパイプライン）向けのエントリポイントです。コマンドライン引数をパースし、`trainer.py` の `train_model` を呼び出します。

```python
# src/trainer_cli.py
import argparse
from trainer import train_model

def main():
    parser = argparse.ArgumentParser(description="Train penguin model via CLI")
    parser.add_argument("--transformed-train-data", type=str, required=True)
    parser.add_argument("--trained-model", type=str, required=True)
    parser.add_argument("--suffix", type=str, default="_xf")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train_model(
        train_data_path=args.transformed_train_data,
        model_output_path=args.trained_model,
        suffix=args.suffix,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )

if __name__ == "__main__":
    main()
```

##### KFP Component アダプター（components/trainer/src/trainer_component.py）
路線2（KFP Artifact 特化パイプライン）向けのアダプターです。KFP SDK の `Input[Dataset]`, `Output[Model]` を受け取り、内部ロジックを呼び出してメタデータを付与します。

```python
# src/trainer_component.py
from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model

@dsl.component(base_image="asia-northeast1-docker.pkg.dev/.../trainer:latest")
def train_model_op(
    transformed_train_data: Input[Dataset],
    trained_model: Output[Model],
    suffix: str = "_xf",
    n_estimators: int = 100,
    random_state: int = 42,
):
    from trainer import train_model

    meta = train_model(
        train_data_path=transformed_train_data.path,
        model_output_path=trained_model.path,
        suffix=suffix,
        n_estimators=n_estimators,
        random_state=random_state,
    )
    trained_model.metadata["framework"] = meta["framework"]
    trained_model.metadata["algorithm"] = meta["algorithm"]
    trained_model.metadata["n_samples"] = meta["n_samples"]
    trained_model.metadata["n_estimators"] = n_estimators
    trained_model.metadata["random_state"] = random_state
```

##### KFP Component 単体テスト（components/trainer/tests/test_trainer_component.py）
KFP アダプターの「インターフェース定義（Spec）」と「実行・メタデータ付与（Execution）」をクラウド環境なしでローカル高速検証します。

```python
# tests/test_trainer_component.py
from pathlib import Path
from kfp.dsl import Dataset, Model
from components.trainer.src.trainer_component import train_model_op

def test_train_model_op_spec():
    """コンポーネントの入出力インターフェース定義（Spec）を検証"""
    spec = train_model_op.component_spec
    assert "transformed_train_data" in spec.inputs
    assert "trained_model" in spec.outputs
    assert spec.outputs["trained_model"].type == "system.Model"

def test_train_model_op_execution(tmp_path: Path):
    """Component アダプターのローカル直接実行とメタデータ付与を検証"""
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("culmen_length_mm_xf,species_xf\n1.0,0\n")
    model_pkl = tmp_path / "model.pkl"

    train_ds = Dataset(name="transformed_train_data", uri=str(train_csv))
    trained_model = Model(name="trained_model", uri=str(model_pkl))

    train_model_op.python_func(
        transformed_train_data=train_ds,
        trained_model=trained_model,
        suffix="_xf",
        n_estimators=10,
        random_state=42,
    )

    assert model_pkl.exists()
    assert trained_model.metadata["framework"] == "scikit-learn"
    assert trained_model.metadata["algorithm"] == "RandomForestClassifier"
    assert trained_model.metadata["n_samples"] == 1
```

---

### 3.2 コンテナイメージと Docker & uv ビルド設計

コンポーネントの実行実体となる Docker コンテナイメージは、軽量かつ高速にビルドできるよう設計された共通の `Dockerfile` 構成を採用しています。パッケージマネージャーとして uv をコンテナ内外で一貫して活用し、マルチステージビルドによるキャッシュ最適化と、柔軟なエントリポイント設計を実現しています。

#### Dockerfile 構成
各コンポーネントのルートに配置される共通の Dockerfile 実装は以下の通りです。マルチステージビルドにより Astral 公式の uv バイナリを抽出し、Python 3.14 の公式スリムイメージ上にシステム環境として依存関係を高速インストールします。

```dockerfile
# 1. uv バイナリのコピー
FROM ghcr.io/astral-sh/uv:latest AS uv_bin

# 2. 実行用ベースイメージ
FROM python:3.14-slim-bookworm

# uv バイナリを配置
COPY --from=uv_bin /uv /uvx /bin/

# 依存パッケージ定義のコピーとインストール（キャッシュ効率化）
WORKDIR /component
COPY pyproject.toml ./
RUN uv pip install --system --no-cache -r pyproject.toml

# ソースコードのコピー
COPY src/ src/
ENV PYTHONPATH=/component/src

# デフォルト実行コマンド（ENTRYPOINT は固定せず、CMD でヘルプ等のフォールバックを提供）
CMD ["python", "-m", "trainer_cli", "--help"]
```

#### 参照公式ドキュメント
- Astral uv Docker ベストプラクティス: [Using uv in Docker](https://docs.astral.sh/uv/guides/docker/)
- Docker マルチステージビルド: [Multi-stage builds](https://docs.docker.com/build/building/multi-stage/)

#### エントリポイント設計の要点と動作
Dockerfile 側で `ENTRYPOINT` を決め打ちせず、KFP パイプライン（または実行者）側にコマンド実行の主導権を持たせることで、同じ 1 つのコンテナイメージで以下のすべてを破綻なく両立します。

- 路線1（汎用コンテナパイプライン: `pipeline_container.py`）:
  `@dsl.container_component` の `command=["python", "-m", "trainer_cli"]` および `args=[...]` により、パイプライン側から明示的に CLI を起動。
- 路線2（KFP Artifact 特化パイプライン: `pipeline_component.py`）:
  `@dsl.component(base_image="...")` により、KFP SDK が自動注入するエグゼキュータコマンド（`python3 -m kfp.dsl.executor_main ...`）が Dockerfile の ENTRYPOINT に邪魔されることなくそのまま正常実行。
- ローカルデバッグ:
  - `docker run image`: デフォルトの `CMD` が実行され、CLI のヘルプが表示される。
  - `docker run -it image bash`: コンテナ内で対話型 bash シェルが問題なく起動し、内部調査が可能。
- ローカルテスト:
  `uv run pytest` により `tests/` 配下の全テストをインメモリで即座に実行可能。

---

### 3.3 KFP コンポーネントアダプター（`*_op`）の実装上の制約と設計判断

KFP SDK v2 を用いて Python 関数をコンポーネント化する際、一般的な Python アプリケーション開発の慣例とは異なる特有の実装パターンが必要となります。ここでは、インポートのスコープとアーティファクトの引数受け渡しという 2 つの重要な設計判断について、その背景と技術的根拠を整理します。

#### 関数内インポート（Inside-Function Import）を採用する技術的理由
Python の一般的なコーディング規約（PEP 8）ではモジュール冒頭でのインポートが推奨されますが、KFP v2 の `@dsl.component` アダプターでは関数内部でのインポート（`from data_generator import generate_data` 等）を採用しています。これには KFP のアーキテクチャに起因する 2 つの決定的な理由があります。

##### KFP の AST シリアライズと一時スクリプト実行モデル
KFP v2 はパイプラインのコンパイル時、`@dsl.component` で修飾された関数の AST（抽象構文木）および関数本体のソースコードのみを抽出して Pipeline Spec（IR YAML）に埋め込みます。
実行時、コンテナ内でエグゼキュータ（`kfp.dsl.executor_main`）が起動すると、YAML 内の関数コードを一時ファイル（`ephemeral_component.py`）としてディスクに書き出し、その関数単体を隔離して実行します。
そのため、関数の外側（モジュール先頭）に記述された import 文やグローバル変数は一時ファイルに持ち込まれず、実行時に `NameError` となります。関数内部に import を書くことで、一時ファイル実行時にも必要な依存モジュールが確実に解決されます。

##### パイプライン定義環境（ホスト環境）のパス汚染防止
もしモジュール先頭に `from data_generator import generate_data` と記述した場合、パイプライン作成者が `pipeline_component.py` で `from components.data_generator import data_generator_op` をインポートした瞬間に、ホストマシンのルート `sys.path` から `data_generator` を探索してしまい、`ModuleNotFoundError` が発生します。
インポートを関数内にカプセル化することで、ホストマシン側でのパイプライン定義・コンパイル処理と、コンテナ内部での実行時ランタイム環境（`PYTHONPATH=/component/src`）を完全に疎結合に保つことができます。

#### 出力先が戻り値（return）ではなく引数（`Output[Dataset]` 等）として渡される理由
通常の Python 関数は計算結果を `return output_data` のように戻り値として返しますが、KFP v2 のコンポーネントでは `train_data: Output[Dataset]` や `trained_model: Output[Model]` のように出力先オブジェクトが関数の引数として渡されます。この設計を採用している理由は以下の 4 点です。

##### オーケストレータによるストレージパスの事前一意割り当て
Vertex AI Pipelines や Kubeflow Pipelines は、パイプラインの実行ごとに Cloud Storage（GCS）等の永続ストレージ上に一意なディレクトリ（例: `gs://bucket/root/run_id/task_id/train_data/`）を事前に決定・割り当てます。
この割り当てられたパス情報が `train_data.path` としてコンポーネントに注入されるため、コンポーネント側はストレージの場所や命名規則を意識することなく、渡された `.path` に対して書き込むだけで自動的にアーティファクトの永続化が完了します。

##### 大規模データ・モデルのストリーミング書き出しと OOM 回避
ML パイプラインで扱うデータセットやモデルは、数百 MB から数百 GB、数 TB に及ぶことがあります。
もし関数の戻り値（`return`）でデータを渡す設計にした場合、全データを一度 Python プロセスのメモリ上に保持・シリアライズしなければならず、コンテナの Out-Of-Memory（OOM）クラッシュを引き起こします。
引数としてディスクパス（`.path`）が渡されることで、pandas や pickle、TensorFlow、PyTorch のストレージ書き出し API を直接呼び出し、ディスクや GCS に直接ストリーミング保存することが可能になります。

##### 実行中におけるリッチなメタデータの動的付与（Metadata Logging）
`Output[Artifact]`（`Dataset`, `Model`, `Metrics`, `ClassificationMetrics` 等）は、単なるファイルパスのラッパーではなく、Vertex AI ML Metadata（MLMD）に登録される第一級オブジェクトです。
引数として受け取ることにより、処理の途中で以下のようなリッチなメタデータや可視化データを直接書き込むことができます：

- `train_data.metadata["rows"] = 1000`（行数の記録）
- `trained_model.metadata["framework"] = "scikit-learn"`（ML フレームワーク情報の記録）
- `metrics.log_metric("accuracy", 0.95)`（評価指標の記録）
- `classification_metrics.log_confusion_matrix(labels, matrix)`（Vertex AI UI 上での対話的混同行列の直接レンダリング）

##### パイプライン DAG（有向非巡回グラフ）の静的リネージ確定
パイプライン定義時（`pipeline_component.py`）、`gen_step.outputs["train_data"]` を後続タスクの引数に渡すことで、KFP コンパイラは各タスク間のデータ依存関係（リネージ・DAG）を静的に解析・確定できます。これにより、パイプラインが実行される前であっても、実行グラフとアーティファクトのフローが完全に可視化・検証可能になります。

---

### 3.4 実装上の考慮事項: コンテナ処理の3大責務と設計バランス

Kubeflow Pipelines におけるコンテナ内部での処理内容は、概念的に次の 3 つに大別されます。

1. ストレージからの読み込み
Cloud Storage やローカルファイルシステムに配置された入力データ（CSV、モデルバイナリ等）を開き、Python のインメモリデータ構造（pandas DataFrame、NumPy 配列、オブジェクト等）へと読み込む処理です。

2. 読み込んだデータを用いた処理
外部ストレージやファイルパスに依存せず、メモリ上に展開されたデータに対して前処理、特徴量変換、モデル学習、メトリクス評価等のアルゴリズム演算を実行する純粋なドメインロジックです。

3. 処理の結果のストレージへの書き込み
演算によって得られた結果オブジェクト（変換後データ、学習済みモデル、評価指標、混同行列等）をシリアライズし、指定された出力先ストレージへ永続化する処理です。

#### 3層分離によるテスト容易性の向上
これら 3 つの処理を関数レベルで明確に分離して記述することには、大きなメリットがあります。
特に「読み込んだデータを用いた処理」をファイル I/O から完全に独立した純粋関数として実装すれば、単体テストにおいてディスクやファイルシステム（一時ディレクトリの作成やモック）を一切介することなく、インメモリのデータのみを用いて高速かつ決定論的にアルゴリズムの検証が行えます。また、将来的にストレージの仕様が変更された場合でも、演算ロジックに影響を与えることなく入出力関数のみを安全に差し替えることが可能になります。

#### 本実装における設計判断（過度な複雑化の回避）
一方で、本プロジェクトで扱う各コンポーネント（`data_generator.py`, `transform.py`, `trainer.py`, `evaluator.py`）は、処理ロジック全体が 20〜40 行程度と非常に簡潔です。
このような小規模かつ自明な処理に対して、読み込み・演算・書き込みを複数の関数やクラスへと厳格に細分化した場合、引数のバケツリレーや中間関数のボイラープレートが増加し、コード全体の記述が過度に複雑化して見通しを損ねてしまいます。
そのため、本実装では関数の過度な細分化をあえて避け、入力パスの読み込みからモデル演算・出力パスへの保存までを 1 つの関数の中で上から下へ素直に完結させることで、コードのシンプルさと直感的な可読性を最優先するバランスを選択しています。

---

## 4. パイプライン設計（2系統のオーケストレーション）

本リポジトリでは、ユーザーが自身の用途や既存資産の状況に合わせて最適なパターンを選択できるよう、目的の異なる 2 系統のパイプライン定義ファイル（`pipeline_container.py` および `pipeline_component.py`）を提供しています。それぞれの対象読者、構成方式、および学べる設計パターンの違いは以下の通りです。

### 4.1 路線1: 汎用コンテナパイプライン（pipeline_container.py）

本路線は、既存の Dockerfile 資産や外部 CLI ツールを最小限の手間で Vertex Pipelines 上に統合したい開発者を対象としています（目的1: シンプル・応用重視）。
各コンポーネントの `src/{name}_cli.py` を呼び出す Container Component（`@dsl.container_component`）を採用しており、コンテナ内部の実装言語やフレームワークに依存せず、コマンドライン引数経由でデータパスやハイパーパラメータを受け渡す汎用的な連携手法を習得できます。

```python
# pipeline_container.py（構成例）
from kfp import dsl

@dsl.container_component
def trainer_container_op(
    transformed_train_data: dsl.Input[dsl.Dataset],
    trained_model: dsl.Output[dsl.Model],
    suffix: str = "_xf",
    n_estimators: int = 100,
    random_state: int = 42,
):
    return dsl.ContainerSpec(
        image="asia-northeast1-docker.pkg.dev/.../trainer:latest",
        command=["python", "-m", "trainer_cli"],
        args=[
            "--transformed-train-data", transformed_train_data.path,
            "--trained-model", trained_model.path,
            "--suffix", suffix,
            "--n-estimators", str(n_estimators),
            "--random-state", str(random_state),
        ],
    )
```

### 4.2 路線2: KFP Artifact 特化パイプライン（pipeline_component.py）

本路線は、KFP SDK v2 の第一級型システムや Vertex AI Pipelines の可視化・追跡機能を最大限に活用したい開発者を対象としています（目的2: KFP SDK Artifacts フル活用）。
各コンポーネントパッケージのファサード（`components.<name>`）から公開されたコンポーネント関数（`*_op`）を直接インポートしてパイプライン DAG を構築します。`Dataset`, `Model`, `Metrics`, `ClassificationMetrics` といった第一級アーティファクトの受け渡しや、動的なメタデータ記録、Vertex AI UI 上での対話的な混同行列の描画手法を網羅的に習得できます。

```python
# pipeline_component.py（構成例）
from kfp import dsl
from components.data_generator import data_generator_op
from components.transform import transform_op
from components.trainer import train_model_op
from components.evaluator import evaluate_model_op

@dsl.pipeline(
    name="penguin-classification-component-pipeline",
    description="Penguin classification pipeline using native KFP components and artifacts",
)
def penguin_component_pipeline(
    dataset_uri: str = "...",
    n_estimators: int = 100,
    random_state: int = 42,
):
    gen_step = data_generator_op(dataset_uri=dataset_uri)
    xf_step = transform_op(input_data=gen_step.outputs["generated_data"])
    train_step = train_model_op(
        transformed_train_data=xf_step.outputs["transformed_train_data"],
        n_estimators=n_estimators,
        random_state=random_state,
    )
    eval_step = evaluate_model_op(
        transformed_test_data=xf_step.outputs["transformed_test_data"],
        model_input=train_step.outputs["trained_model"],
    )
```

### 4.3 評価メトリクスと混同行列可視化の設計（ClassificationMetrics の制約と棲み分け）

モデル評価ステップ（Evaluator）における評価指標および混同行列（Confusion Matrix）の出力は、Container Pipeline と Component Pipeline の設計思想の違いが最も色濃く現れる部分です。

#### 1. KFP v2 における ClassificationMetrics の動作メカニズム
KFP v2 で Vertex AI コンソール上に対話的な混同行列ウィジェットを描画するには、`kfp.dsl.ClassificationMetrics` 型の出力アーティファクトに対して `classification_metrics.log_confusion_matrix(categories, matrix)` を呼び出す必要があります。
このメソッドは内部的に、Vertex AI MLMD（機械学習メタデータ）が解釈可能なメタデータ構造（`{"confusionMatrix": {"annotationSpecs": [...], "rows": [...]}}`）を構築し、アーティファクトのメタデータ辞書に格納します。

#### 2. CLI / Container Component における制約事項
`@dsl.container_component` を用いた Container Pipeline では、以下の構造的制約が存在します：
- 【引数がファイルパス文字列である点】:
  CLI エントリポイント（`evaluator_cli.py`）がターミナル引数として受け取るのは `--confusion-matrix-path /path/to/confusion_matrix` という単なるファイルパス文字列（`str`）です。Python インスタンスとしての `ClassificationMetrics` オブジェクトが存在しないため、`log_confusion_matrix()` メソッドを直接呼び出すことができません。
- 【フレームワーク独立性（ポータビリティ）とのトレードオフ】:
  CLI 側で混同行列ウィジェットを描画させようとする場合、CLI スクリプト内で KFP SDK をインポートしてオブジェクトを自前でインスタンス化するか、Vertex AI 固有の内部 JSON スキーマを手動で組み立てて出力ファイルに書き出す必要があります。しかしこれを行うと、「KFP やクラウド環境に依存せず、Docker 単体やローカルでも動く純粋な Python スクリプトである」という CLI 本来のポータビリティが破壊されてしまいます。

#### 3. 本リポジトリにおける設計判断と棲み分け
以上のトレードオフを踏まえ、本リポジトリでは以下のように明確な役割分担を行っています：
- 【Container Pipeline（`pipeline_container.py` / CLI 方式）】:
  コンテナおよび CLI のポータビリティを最優先とし、標準的な画像ファイル（matplotlib による PNG 形式）および汎用 Metrics JSON を出力します。Docker 単体で実行した場合でも画像ビューア等で結果を即座に確認可能です。
- 【Component Pipeline（`pipeline_component.py` / ネイティブ方式）】:
  KFP v2 の型システムと Vertex AI Pipelines の UI 統合を最優先とし、`Output[ClassificationMetrics]` と `log_confusion_matrix()` を使用して、Vertex AI コンソール上にリッチな対話的混同行列ウィジェットを直接レンダリングします。

### 4.4 パイプラインコンパイルと IR YAML の生成仕様
KFP SDK v2 の `kfp.compiler.Compiler` を用いて、Python で定義したパイプライン DAG を Vertex AI / KFP 互換の Intermediate Representation（IR YAML）仕様ファイルへとコンパイルします。

`pipeline_container.py` をコンパイルすると `kfp_container_pipeline.yaml` が生成され、`pipeline_component.py` をコンパイルすると `kfp_component_pipeline.yaml` が生成されます。

コンパイル処理はローカル環境（`uv run python pipeline_component.py`）で即座に実行可能であり、クラウド環境へアクセスすることなくパイプライン DAG と引数定義の整合性を検証できます。

---

## 5. デプロイおよび実行仕様（ビルド・登録・Vertex AI Pipelines 投入）

本パイプラインのデプロイメントライフサイクルは、「Google Cloud インフラストラクチャの準備」「コンテナイメージのクロスプラットフォームビルド & レジストリ登録」「パイプラインのコンパイル」「Vertex AI Pipelines への投入（PipelineJob の実行）」という 4 つの明確なステップから構成されます。

### 5.1 クラウドインフラストラクチャ準備（GCS / Artifact Registry / IAM）

パイプラインのデプロイおよび実行に先立ち、Google Cloud 上で以下のリソースと権限をプロビジョニングします。

#### 1. Cloud Storage（GCS）バケット（Pipeline Root）
パイプラインの各ステップが出力するアーティファクト（データセット、前処理済みデータ、学習済みモデル、評価指標）を保管するバケットを作成します。
```bash
gcloud storage buckets create gs://${BUCKET_NAME} --location=asia-northeast1
```

#### 2. Google Cloud Artifact Registry
各コンポーネントの Docker コンテナイメージを格納する Docker 形式のレジストリを作成します。従来の Container Registry（`*.gcr.io`）は非推奨化されているため、Artifact Registry の使用が必須です。
```bash
gcloud artifacts repositories create pipeline-components \
    --repository-format=docker \
    --location=asia-northeast1 \
    --description="Docker repository for KFP pipeline components"
```

#### 3. Vertex AI Pipelines サービスアカウントと IAM 権限
Vertex AI Pipelines のワーカーノードが GCS 上のアーティファクトへアクセスし、Artifact Registry からコンテナイメージを pull するために必要な IAM ロールを付与します。
- `Vertex AI ユーザー` (`roles/aiplatform.user`)
- `ストレージオブジェクト管理者` (`roles/storage.objectAdmin`)
- `Artifact Registry 読み取り` (`roles/artifactregistry.reader`)

### 5.2 コンテナイメージのビルドと Artifact Registry 登録

Vertex AI Pipelines の実行基盤は Intel / AMD x86_64（`linux/amd64`）アーキテクチャで稼働しています。開発者が Apple Silicon Mac（ARM64）等で開発を行っている場合、プラットフォームの不一致による起動時エラー（`exec format error`）を防ぐため、確実に `linux/amd64` 向けバイナリとしてビルドして Artifact Registry へ push する必要があります。

#### 戦略1: ローカルクロスビルド（Docker Buildx による linux/amd64 指定）
Docker Desktop の BuildKit / QEMU エミュレーションを活用し、開発マシンから直接 Intel 向けコンテナをビルドして push します。
```bash
docker buildx build --platform linux/amd64 --no-cache \
    -t asia-northeast1-docker.pkg.dev/${PROJECT_ID}/pipeline-components/${COMPONENT_NAME}:latest \
    --push components/${COMPONENT_NAME}
```
本プロジェクトでは `deploy_all_component.sh` スクリプトにより、全コンポーネント（data_generator, transform, trainer, evaluator）のクロスビルドと push を一括実行できます。

#### 戦略2: クラウドネイティブビルド（Google Cloud Build によるリモートビルド）
ローカル Docker デーモンや QEMU エミュレーションに依存せず、Google Cloud 上のネイティブ x86_64 ワーカー環境でビルドを実行します。ローカルマシンの負荷が皆無であり、CI/CD パイプラインでの自動ビルドに最適です。
```bash
gcloud builds submit \
    --tag asia-northeast1-docker.pkg.dev/${PROJECT_ID}/pipeline-components/${COMPONENT_NAME}:latest \
    components/${COMPONENT_NAME}
```

### 5.3 パイプラインコンパイルと YAML 生成

各コンポーネントのイメージ登録後、対象パイプラインをコンパイルして実行仕様書（IR YAML）を出力します。

```bash
# 路線1（汎用コンテナパイプライン）のコンパイル
uv run python pipeline_container.py

# 路線2（KFP Artifact 特化パイプライン）のコンパイル
uv run python pipeline_component.py
```

### 5.4 Vertex AI Pipelines への投入と実行（PipelineJob）

コンパイルされた YAML ファイルを指定し、Google Cloud Vertex AI SDK（`google.cloud.aiplatform`）を用いてパイプラインジョブを Vertex Pipelines へ投入します。

```python
from google.cloud import aiplatform

aiplatform.init(
    project="your-gcp-project-id",
    location="asia-northeast1",
)

job = aiplatform.PipelineJob(
    display_name="penguin-classification-run",
    template_path="kfp_component_pipeline.yaml",
    pipeline_root="gs://your-pipeline-root-bucket/pipeline_root",
    parameter_values={
        "n_estimators": 100,
        "random_state": 42,
    },
    enable_caching=True,
)

job.submit()
```

#### 実行監視と Vertex AI UI での可視化
ジョブが投入されると、Vertex AI コンソールの Pipelines ダッシュボード上で以下の情報がリアルタイムに確認可能となります。
- DAG の視覚的な実行進行状況
- 各ステップの入出力アーティファクトと Cloud Storage パス
- Vertex AI ML Metadata（MLMD）による実行パラメータとメタデータ（`rows`, `framework`, `n_samples` 等）
- Evaluator コンポーネントが出力した対話的混同行列（Confusion Matrix）のグラフィカル表示

### 5.5 参照公式ドキュメント
- Vertex AI パイプライン実行ガイド: [Run a pipeline](https://cloud.google.com/vertex-ai/docs/pipelines/run-pipeline)
- Vertex AI パイプラインジョブ Python クライアント: [PipelineJob API reference](https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.PipelineJob)
- Artifact Registry Docker ガイド: [Push and pull Docker images](https://cloud.google.com/artifact-registry/docs/docker/pushing-and-pulling)
- Google Cloud Build 概要: [Cloud Build Overview](https://cloud.google.com/build/docs/overview)
