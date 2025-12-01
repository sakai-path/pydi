"""
PyDI + FastAPI 統合例

実際のWebアプリケーションでDIコンテナを使用する例
"""

from fastapi import FastAPI, Depends, Request
from contextlib import asynccontextmanager
from abc import ABC, abstractmethod
from typing import Generator
import uvicorn

# PyDIからインポート（同じディレクトリにpydi_container.pyがある前提）
from pydi_container import (
    Container, ContainerBuilder, Lifetime,
    injectable, singleton, scoped, transient
)

# =============================================================================
# インターフェース定義
# =============================================================================

class ILogger(ABC):
    @abstractmethod
    def info(self, message: str) -> None: ...
    
    @abstractmethod
    def error(self, message: str) -> None: ...

class IUserRepository(ABC):
    @abstractmethod
    def get_user(self, user_id: int) -> dict: ...
    
    @abstractmethod
    def create_user(self, name: str, email: str) -> dict: ...

class IUserService(ABC):
    @abstractmethod
    def get_user_profile(self, user_id: int) -> dict: ...

# =============================================================================
# 実装クラス
# =============================================================================

@singleton
class ConsoleLogger(ILogger):
    def __init__(self):
        print("🔵 ConsoleLogger: 初期化 (SINGLETON)")
    
    def info(self, message: str) -> None:
        print(f"[INFO] {message}")
    
    def error(self, message: str) -> None:
        print(f"[ERROR] {message}")

@scoped
class InMemoryUserRepository(IUserRepository):
    """スコープ付き - リクエストごとに新しいインスタンス"""
    
    _instance_counter = 0
    
    def __init__(self, logger: ILogger):
        InMemoryUserRepository._instance_counter += 1
        self._id = InMemoryUserRepository._instance_counter
        self._logger = logger
        self._users = {
            1: {"id": 1, "name": "Alice", "email": "alice@example.com"},
            2: {"id": 2, "name": "Bob", "email": "bob@example.com"},
        }
        self._logger.info(f"🟠 UserRepository #{self._id}: 初期化 (SCOPED)")
    
    def get_user(self, user_id: int) -> dict:
        self._logger.info(f"UserRepository #{self._id}: get_user({user_id})")
        return self._users.get(user_id, {"error": "not found"})
    
    def create_user(self, name: str, email: str) -> dict:
        new_id = max(self._users.keys()) + 1
        user = {"id": new_id, "name": name, "email": email}
        self._users[new_id] = user
        self._logger.info(f"UserRepository #{self._id}: created user {new_id}")
        return user
    
    def dispose(self) -> None:
        self._logger.info(f"🟠 UserRepository #{self._id}: 破棄")

@transient
class UserService(IUserService):
    """トランジェント - 毎回新しいインスタンス"""
    
    _instance_counter = 0
    
    def __init__(self, user_repo: IUserRepository, logger: ILogger):
        UserService._instance_counter += 1
        self._id = UserService._instance_counter
        self._user_repo = user_repo
        self._logger = logger
        self._logger.info(f"🟣 UserService #{self._id}: 初期化 (TRANSIENT)")
    
    def get_user_profile(self, user_id: int) -> dict:
        self._logger.info(f"UserService #{self._id}: get_user_profile({user_id})")
        user = self._user_repo.get_user(user_id)
        if "error" in user:
            return user
        return {
            "profile": user,
            "service_instance": self._id,
        }

# =============================================================================
# DIコンテナのセットアップ
# =============================================================================

def create_container() -> Container:
    """アプリケーション用のDIコンテナを構築"""
    return (ContainerBuilder()
        .add_singleton(ILogger, ConsoleLogger)
        .add_scoped(IUserRepository, InMemoryUserRepository)
        .add_transient(IUserService, UserService)
        .build())

# グローバルコンテナ
container: Container = None

# =============================================================================
# FastAPI アプリケーション
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションのライフサイクル管理"""
    global container
    print("=" * 50)
    print("🚀 アプリケーション起動")
    print("=" * 50)
    
    # コンテナ初期化
    container = create_container()
    
    yield
    
    # シャットダウン時のクリーンアップ
    print("=" * 50)
    print("👋 アプリケーション終了")
    print("=" * 50)

app = FastAPI(
    title="PyDI + FastAPI Demo",
    description="依存性注入コンテナを使用したFastAPIアプリケーション",
    lifespan=lifespan
)

# =============================================================================
# 依存性注入のためのFastAPI Depends
# =============================================================================

def get_scope():
    """リクエストスコープを作成"""
    with container.scope() as scope:
        yield scope

def get_user_service(scope = Depends(get_scope)) -> IUserService:
    """UserServiceを解決"""
    return container.resolve(IUserService)

def get_logger() -> ILogger:
    """Loggerを解決（シングルトンなのでスコープ不要）"""
    return container.resolve(ILogger)

# =============================================================================
# エンドポイント
# =============================================================================

@app.get("/")
def root():
    return {
        "message": "PyDI + FastAPI Demo",
        "endpoints": {
            "/users/{user_id}": "ユーザー情報を取得",
            "/demo/lifetime": "ライフタイムの違いをデモ",
        }
    }

@app.get("/users/{user_id}")
def get_user(
    user_id: int,
    user_service: IUserService = Depends(get_user_service)
):
    """ユーザー情報を取得"""
    return user_service.get_user_profile(user_id)

@app.get("/demo/lifetime")
def demo_lifetime():
    """
    ライフタイムの違いを確認するデモエンドポイント
    
    同じリクエスト内で複数回サービスを解決して、
    インスタンスIDの違いを確認できます。
    """
    results = []
    
    with container.scope():
        # 1回目の解決
        svc1 = container.resolve(IUserService)
        results.append({
            "resolution": 1,
            "user_service_id": svc1._id,
            "user_repo_id": svc1._user_repo._id,
        })
        
        # 2回目の解決
        svc2 = container.resolve(IUserService)
        results.append({
            "resolution": 2,
            "user_service_id": svc2._id,
            "user_repo_id": svc2._user_repo._id,
        })
        
        # 3回目の解決
        svc3 = container.resolve(IUserService)
        results.append({
            "resolution": 3,
            "user_service_id": svc3._id,
            "user_repo_id": svc3._user_repo._id,
        })
    
    return {
        "explanation": {
            "UserService": "TRANSIENT - 毎回異なるID",
            "UserRepository": "SCOPED - 同じスコープ内は同じID",
            "Logger": "SINGLETON - 常に同じインスタンス",
        },
        "results": results
    }

@app.get("/demo/scope-comparison")
def demo_scope_comparison():
    """
    異なるスコープでの解決を比較
    """
    results = []
    
    # スコープ1
    with container.scope():
        svc = container.resolve(IUserService)
        results.append({
            "scope": 1,
            "user_service_id": svc._id,
            "user_repo_id": svc._user_repo._id,
        })
    
    # スコープ2（新しいスコープ）
    with container.scope():
        svc = container.resolve(IUserService)
        results.append({
            "scope": 2,
            "user_service_id": svc._id,
            "user_repo_id": svc._user_repo._id,
        })
    
    return {
        "note": "UserRepositoryのIDがスコープごとに変わることを確認",
        "results": results
    }

# =============================================================================
# 実行
# =============================================================================

if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════╗
║         PyDI + FastAPI Integration Demo                       ║
╠═══════════════════════════════════════════════════════════════╣
║  http://localhost:8000          - API Root                    ║
║  http://localhost:8000/docs     - Swagger UI                  ║
║  http://localhost:8000/users/1  - Get User                    ║
║  http://localhost:8000/demo/lifetime - Lifetime Demo          ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    uvicorn.run(app, host="0.0.0.0", port=8000)
