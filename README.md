# PyDI - Python Dependency Injection Container

メタプログラミングを駆使した型安全な依存性注入コンテナ

## 🚀 クイックスタート

### 必要環境

- Python 3.10+

### インストール

```bash
# リポジトリをクローン
git clone https://github.com/your-username/pydi.git
cd pydi

# 依存関係をインストール（デモ用）
pip install -r requirements.txt
```

### 基本的な使い方

```python
from pydi_container import Container, ContainerBuilder, Lifetime
from abc import ABC, abstractmethod

# 1. インターフェース定義
class ILogger(ABC):
    @abstractmethod
    def log(self, message: str) -> None: ...

class IUserService(ABC):
    @abstractmethod
    def get_user(self, user_id: int) -> dict: ...

# 2. 実装クラス
class ConsoleLogger(ILogger):
    def log(self, message: str) -> None:
        print(f"[LOG] {message}")

class UserService(IUserService):
    def __init__(self, logger: ILogger):  # 型ヒントで依存を宣言
        self._logger = logger
    
    def get_user(self, user_id: int) -> dict:
        self._logger.log(f"Getting user {user_id}")
        return {"id": user_id, "name": f"User_{user_id}"}

# 3. コンテナ構築
container = (ContainerBuilder()
    .add_singleton(ILogger, ConsoleLogger)
    .add_transient(IUserService, UserService)
    .build())

# 4. サービス解決
with container.scope():
    user_service = container.resolve(IUserService)
    print(user_service.get_user(42))
```

## 🎮 デモを試す

### Streamlit インタラクティブデモ

```bash
streamlit run streamlit_demo.py
```

ブラウザで http://localhost:8501 を開くと：
- 📊 依存グラフの可視化
- 🔄 循環依存の検出デモ
- ⏰ ライフタイムの比較
- 🎮 カスタム依存関係のテスト

### FastAPI 統合例

```bash
python fastapi_example.py
```

ブラウザで http://localhost:8000/docs を開くとSwagger UIで確認できます。

## 📚 ライフタイム

| ライフタイム | 説明 | 用途例 |
|:--|:--|:--|
| `TRANSIENT` | 毎回新規インスタンス | バリデータ、ステートレスサービス |
| `SINGLETON` | アプリ全体で1つ | ロガー、設定、コネクションプール |
| `SCOPED` | スコープ内で1つ | DBコネクション、リクエストコンテキスト |

## 🛠 機能

- ✅ コンストラクタインジェクション（型ヒントから自動解決）
- ✅ プロパティインジェクション（`Inject`ディスクリプタ）
- ✅ 3種類のライフタイム管理
- ✅ 循環依存の自動検出
- ✅ ライフタイム不整合の警告
- ✅ 非同期初期化サポート
- ✅ スレッドセーフ
- ✅ 弱参照によるメモリリーク防止

## 📁 ファイル構成

```
pydi/
├── pydi_container.py      # メインのDIコンテナ実装
├── test_advanced_features.py  # 高度な機能のテスト
├── streamlit_demo.py      # Streamlitインタラクティブデモ
├── fastapi_example.py     # FastAPI統合例
├── requirements.txt       # 依存関係
├── qiita_article.md       # Qiita記事
└── README.md              # このファイル
```

## 📖 高度な使い方

### プロパティインジェクション

```python
from pydi_container import Inject

class Controller:
    logger = Inject(ILogger)  # 遅延解決される
    
    def handle(self):
        self.logger.log("Handling request")  # ここで初めて解決
```

### ファクトリパターン

```python
def config_factory() -> Config:
    env = os.environ.get("APP_ENV", "dev")
    return Config(env)

container = (ContainerBuilder()
    .add_singleton(IConfig, factory=config_factory)
    .build())
```

### 非同期初期化

```python
class DatabaseConnection:
    async def initialize_async(self):
        await self.connect()

async with container.scope_async():
    db = await container.resolve_async(DatabaseConnection)
```

## 📝 License

MIT License
