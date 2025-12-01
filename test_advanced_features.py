"""
PyDI 高度な機能テスト

このファイルでは以下をデモンストレーション:
1. 循環依存の検出
2. ライフタイム不整合の検出
3. ファクトリパターン
4. 複雑な依存グラフ
"""

from pydi_container import (
    Container, ContainerBuilder, Lifetime,
    ServiceDescriptor, CircularDependencyError, LifetimeMismatchError,
    injectable, singleton, scoped, transient, Inject,
    DependencyGraph
)
from abc import ABC, abstractmethod
from typing import Optional


# =============================================================================
# テスト1: 循環依存の検出
# =============================================================================

print("=" * 70)
print("テスト1: 循環依存の検出")
print("=" * 70)

class ServiceA:
    """ServiceB に依存"""
    def __init__(self, b: 'ServiceB') -> None:
        self.b = b

class ServiceB:
    """ServiceC に依存"""
    def __init__(self, c: 'ServiceC') -> None:
        self.c = c

class ServiceC:
    """ServiceA に依存 → 循環！"""
    def __init__(self, a: 'ServiceA') -> None:
        self.a = a

try:
    container = (ContainerBuilder()
        .add_transient(ServiceA)
        .add_transient(ServiceB)
        .add_transient(ServiceC)
        .build())
except CircularDependencyError as e:
    print(f"✓ 循環依存を検出: {e}")
    print(f"  循環チェーン: {[t.__name__ for t in e.chain]}")


# =============================================================================
# テスト2: ライフタイム不整合の検出（Captive Dependency）
# =============================================================================

print("\n" + "=" * 70)
print("テスト2: ライフタイム不整合の検出（Captive Dependency問題）")
print("=" * 70)

class IDatabase(ABC):
    @abstractmethod
    def query(self, sql: str) -> list: ...

class IDatabaseConnection(ABC):
    @abstractmethod
    def execute(self, sql: str) -> None: ...

@scoped  # スコープ付き（リクエストごとに新規）
class DatabaseConnection(IDatabaseConnection):
    def execute(self, sql: str) -> None:
        print(f"  Executing: {sql}")

@singleton  # シングルトン（アプリ全体で1つ）
class DatabaseService(IDatabase):
    """
    危険！シングルトンがスコープ付きサービスに依存
    → 最初のスコープのConnectionがずっと使われる問題
    """
    def __init__(self, conn: IDatabaseConnection) -> None:
        self.conn = conn
    
    def query(self, sql: str) -> list:
        self.conn.execute(sql)
        return []

# ライフタイム検証
container = Container()
container.register_scoped(IDatabaseConnection, DatabaseConnection)
container.register_singleton(IDatabase, DatabaseService)

warnings = container.validate(strict=False)
if warnings:
    print("✓ ライフタイム不整合を検出:")
    for w in warnings:
        print(f"  - {w}")
else:
    print("警告なし")


# =============================================================================
# テスト3: ファクトリパターン
# =============================================================================

print("\n" + "=" * 70)
print("テスト3: ファクトリパターンによる動的生成")
print("=" * 70)

class IConfig(ABC):
    @abstractmethod
    def get(self, key: str) -> str: ...

class Config(IConfig):
    def __init__(self, env: str) -> None:
        self.env = env
        self._data = {
            "dev": {"db": "localhost", "debug": "true"},
            "prod": {"db": "prod-server", "debug": "false"}
        }
    
    def get(self, key: str) -> str:
        return self._data.get(self.env, {}).get(key, "")
    
    def __repr__(self) -> str:
        return f"Config(env={self.env})"

# ファクトリ関数で環境に応じた設定を作成
import os
def config_factory() -> Config:
    env = os.environ.get("APP_ENV", "dev")
    print(f"  ファクトリ呼び出し: 環境={env}")
    return Config(env)

container = (ContainerBuilder()
    .add_singleton(IConfig, factory=config_factory)
    .build())

with container.scope():
    config1 = container.resolve(IConfig)
    config2 = container.resolve(IConfig)
    print(f"  config1: {config1}")
    print(f"  config1 is config2: {config1 is config2}  (シングルトンなので同一)")


# =============================================================================
# テスト4: 複雑な依存グラフとトポロジカルソート
# =============================================================================

print("\n" + "=" * 70)
print("テスト4: 複雑な依存グラフの可視化")
print("=" * 70)

class ILogger(ABC):
    pass

class ICache(ABC):
    pass

class IValidator(ABC):
    pass

class IProcessor(ABC):
    pass

class IOrchestrator(ABC):
    pass

class Logger(ILogger):
    pass

class RedisCache(ICache):
    def __init__(self, logger: ILogger) -> None:
        pass

class DataValidator(IValidator):
    def __init__(self, logger: ILogger) -> None:
        pass

class DataProcessor(IProcessor):
    def __init__(self, cache: ICache, validator: IValidator, logger: ILogger) -> None:
        pass

class Orchestrator(IOrchestrator):
    def __init__(self, processor: IProcessor, cache: ICache, logger: ILogger) -> None:
        pass

# 依存グラフを構築
graph = DependencyGraph()
graph.add_service(ILogger, set(), Lifetime.SINGLETON)
graph.add_service(ICache, {ILogger}, Lifetime.SINGLETON)
graph.add_service(IValidator, {ILogger}, Lifetime.TRANSIENT)
graph.add_service(IProcessor, {ICache, IValidator, ILogger}, Lifetime.SCOPED)
graph.add_service(IOrchestrator, {IProcessor, ICache, ILogger}, Lifetime.TRANSIENT)

# 循環依存チェック
cycle = graph.detect_cycles()
print(f"循環依存: {'なし' if cycle is None else cycle}")

# 初期化順序（トポロジカルソート）
order = graph.topological_sort()
print(f"初期化順序: {[t.__name__ for t in order]}")


# =============================================================================
# テスト5: プロパティインジェクション（遅延解決）
# =============================================================================

print("\n" + "=" * 70)
print("テスト5: プロパティインジェクション（Inject ディスクリプタ）")
print("=" * 70)

class SimpleLogger:
    def log(self, msg: str) -> None:
        print(f"  [LOG] {msg}")

class Controller:
    # Injectディスクリプタを使用した遅延注入
    # 注: クラス変数として実際にInjectインスタンスを作成する必要がある
    logger = Inject(SimpleLogger)
    
    def __init__(self) -> None:
        print("  Controller.__init__ 呼び出し（この時点ではloggerは未解決）")
    
    def handle(self) -> None:
        # ここで初めてloggerが解決される
        self.logger.log("handle() が呼ばれました")

container = Container()
container.register_singleton(SimpleLogger)
container.register_transient(Controller)

with container.scope():
    ctrl = container.resolve(Controller)
    ctrl.handle()


# =============================================================================
# テスト6: Optional依存の処理
# =============================================================================

print("\n" + "=" * 70)
print("テスト6: Optional依存の処理")
print("=" * 70)

class IMetrics(ABC):
    @abstractmethod
    def record(self, name: str, value: float) -> None: ...

class ServiceWithOptional:
    def __init__(self, logger: SimpleLogger, metrics: Optional[IMetrics] = None) -> None:
        self._logger = logger
        self._metrics = metrics
        self._logger.log(f"metrics 注入: {self._metrics}")
    
    def do_work(self) -> None:
        self._logger.log("作業実行中...")
        if self._metrics:
            self._metrics.record("work_done", 1.0)

# IMetricsは登録しない → Noneが注入される
container = Container()
container.register_singleton(SimpleLogger)
container.register_transient(ServiceWithOptional)

with container.scope():
    svc = container.resolve(ServiceWithOptional)
    svc.do_work()


# =============================================================================
# テスト7: スレッドセーフ性のデモ
# =============================================================================

print("\n" + "=" * 70)
print("テスト7: マルチスレッドでのシングルトン安全性")
print("=" * 70)

import threading
import time

class Counter:
    _instance_count = 0
    _lock = threading.Lock()
    
    def __init__(self) -> None:
        with Counter._lock:
            Counter._instance_count += 1
            self._id = Counter._instance_count
        time.sleep(0.01)  # 初期化に時間がかかるシミュレーション
    
    @property
    def id(self) -> int:
        return self._id

container = Container()
container.register_singleton(Counter)
container.validate()

instances = []
def resolve_counter():
    inst = container.resolve(Counter)
    instances.append(inst.id)

threads = [threading.Thread(target=resolve_counter) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"  取得されたインスタンスID: {instances}")
print(f"  ユニークID数: {len(set(instances))} (シングルトンなら1)")


print("\n" + "=" * 70)
print("全テスト完了！")
print("=" * 70)
