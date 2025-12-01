"""
PyDI - Pythonメタプログラミングを駆使した型安全な依存性注入コンテナ

高度なPythonテクニック:
- メタクラス (ServiceMeta)
- デコレータ (injectable, singleton, scoped, transient, inject)
- ディスクリプタプロトコル (Inject)
- ジェネリクス・型変数 (TypeVar, Generic)
- inspect によるイントロスペクション
- 弱参照 (WeakValueDictionary)
- コンテキストマネージャ (Scope)
- 非同期処理 (asyncio)
- トポロジカルソート (循環依存検出)
"""

from __future__ import annotations
import asyncio
import inspect
import threading
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
from typing import (
    Any, Callable, Dict, Generic, List, Optional, Set, Tuple, Type, TypeVar,
    Union, get_type_hints, get_origin, get_args, Protocol, runtime_checkable,
    overload, cast, ClassVar
)
from weakref import WeakValueDictionary

# =============================================================================
# バージョン要件チェック
# =============================================================================

import sys
if sys.version_info < (3, 10):
    raise RuntimeError(
        f"PyDI requires Python 3.10+. Current version: {sys.version_info.major}.{sys.version_info.minor}"
    )

# =============================================================================
# 型変数とプロトコル定義
# =============================================================================

T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)
TService = TypeVar('TService')
TImpl = TypeVar('TImpl')

@runtime_checkable
class Disposable(Protocol):
    """
    リソース解放が必要なサービスのプロトコル
    
    typing.Protocol + @runtime_checkable により、
    isinstance() チェックと静的型検査の両方で使用可能
    """
    def dispose(self) -> None: ...

@runtime_checkable
class AsyncDisposable(Protocol):
    """非同期リソース解放が必要なサービスのプロトコル"""
    async def dispose_async(self) -> None: ...

@runtime_checkable
class AsyncInitializable(Protocol):
    """
    非同期初期化が必要なサービスのプロトコル
    
    resolve_async() で解決すると、自動的に initialize_async() が呼ばれる
    """
    async def initialize_async(self) -> None: ...


# =============================================================================
# ライフタイム管理
# =============================================================================

class Lifetime(Enum):
    """サービスのライフタイムスコープ"""
    TRANSIENT = auto()   # 毎回新規インスタンス
    SINGLETON = auto()   # アプリケーション全体で1つ
    SCOPED = auto()      # スコープ内で1つ


@dataclass(frozen=True)
class ServiceDescriptor:
    """サービス登録の記述子（イミュータブル）"""
    service_type: Type[Any]
    implementation_type: Optional[Type[Any]] = None
    factory: Optional[Callable[..., Any]] = None
    instance: Optional[Any] = None
    lifetime: Lifetime = Lifetime.TRANSIENT
    
    def __post_init__(self) -> None:
        # バリデーション: 少なくとも1つの解決方法が必要
        if all(x is None for x in [self.implementation_type, self.factory, self.instance]):
            raise ValueError(
                f"ServiceDescriptor for {self.service_type.__name__} must have "
                "implementation_type, factory, or instance"
            )


# =============================================================================
# 例外クラス
# =============================================================================

class DIException(Exception):
    """DI関連の基底例外"""
    pass

class ServiceNotFoundError(DIException):
    """サービスが見つからない"""
    def __init__(self, service_type: Type[Any]):
        self.service_type = service_type
        super().__init__(f"Service not registered: {service_type.__name__}")

class CircularDependencyError(DIException):
    """循環依存検出"""
    def __init__(self, chain: List[Type[Any]]):
        self.chain = chain
        chain_str = " -> ".join(t.__name__ for t in chain)
        super().__init__(f"Circular dependency detected: {chain_str}")

class ScopeError(DIException):
    """スコープ関連エラー"""
    pass

class LifetimeMismatchError(DIException):
    """ライフタイム不整合（短命が長命に依存）"""
    def __init__(self, consumer: Type[Any], dependency: Type[Any], 
                 consumer_lifetime: Lifetime, dependency_lifetime: Lifetime):
        super().__init__(
            f"Lifetime mismatch: {consumer.__name__} ({consumer_lifetime.name}) "
            f"cannot depend on {dependency.__name__} ({dependency_lifetime.name})"
        )


# =============================================================================
# ディスクリプタ: プロパティインジェクション用
# =============================================================================

class Inject(Generic[T]):
    """
    ディスクリプタプロトコルを使用したプロパティインジェクション
    
    使用例:
        class MyService:
            logger = Inject(ILogger)  # 自動注入される（遅延解決）
    
    注意:
        - Container.resolve() 経由でインスタンス化すると自動的にバインドされる
        - 手動でインスタンス化した場合は bind_container() を呼ぶ必要がある
    """
    
    __slots__ = ('_service_type', '_name', '_container_ref')
    
    def __init__(self, service_type: Optional[Type[T]] = None):
        self._service_type = service_type
        self._name: Optional[str] = None
        self._container_ref: Optional[Callable[[], 'Container']] = None
    
    def __set_name__(self, owner: Type[Any], name: str) -> None:
        """クラス定義時に呼ばれる（Python 3.6+）"""
        self._name = name
        
        # 型ヒントからサービス型を取得
        if self._service_type is None:
            hints = get_type_hints(owner) if hasattr(owner, '__annotations__') else {}
            if name in hints:
                hint = hints[name]
                # Inject[T] から T を抽出
                origin = get_origin(hint)
                if origin is Inject or (isinstance(origin, type) and issubclass(origin, Inject)):
                    args = get_args(hint)
                    if args:
                        self._service_type = args[0]
    
    def __get__(self, obj: Optional[Any], owner: Type[Any]) -> Union['Inject[T]', T]:
        if obj is None:
            return self
        
        # インスタンスにキャッシュされた値があれば返す
        cache_attr = f'_inject_cache_{self._name}'
        if hasattr(obj, cache_attr):
            cached = getattr(obj, cache_attr)
            if cached is not None:
                return cached
        
        # コンテナバインドチェック（改善されたエラーメッセージ）
        if self._container_ref is None:
            raise DIException(
                f"Inject descriptor '{owner.__name__}.{self._name}' is not bound to a container.\n"
                f"解決方法:\n"
                f"  1. Container.resolve({owner.__name__}) を使用してインスタンス化する（推奨）\n"
                f"  2. 手動でインスタンス化した場合は、Inject.bind_container(container) を呼び出す"
            )
        
        container = self._container_ref()
        if container is None:
            raise DIException(
                f"Container has been garbage collected. "
                f"Inject descriptor '{owner.__name__}.{self._name}' が参照していたコンテナが破棄されました。"
            )
        
        if self._service_type is None:
            raise DIException(
                f"Cannot determine service type for {owner.__name__}.{self._name}. "
                f"Inject(ServiceType) のように型を明示してください。"
            )
        
        instance = container.resolve(self._service_type)
        setattr(obj, cache_attr, instance)
        return instance
    
    def __set__(self, obj: Any, value: T) -> None:
        cache_attr = f'_inject_cache_{self._name}'
        setattr(obj, cache_attr, value)
    
    def bind_container(self, container: 'Container') -> None:
        """コンテナへの弱参照をバインド"""
        self._container_ref = weakref.ref(container)


# =============================================================================
# メタクラス: サービスの自動登録と検証
# =============================================================================

class ServiceMeta(type):
    """
    サービスクラス用メタクラス
    - 依存関係の静的解析
    - 自動登録のサポート
    - コンストラクタシグネチャの検証
    
    ⚠️ 制限事項:
    前方参照（文字列アノテーション）を使用したクラス間の相互依存がある場合、
    クラス定義時点では依存関係を解決できません。
    その場合は Container.resolve() 時に inspect.signature で動的解決されます。
    
    例:
        class ServiceA:
            def __init__(self, b: 'ServiceB'): ...  # 前方参照
        
        class ServiceB:
            def __init__(self, a: 'ServiceA'): ...  # 前方参照
        
        # → メタクラスでは依存関係が空と見なされるが、
        #   resolve() 時に正しく解決される
    """
    
    _registry: ClassVar[Dict[Type[Any], 'ServiceDescriptor']] = {}
    
    def __new__(mcs, name: str, bases: Tuple[type, ...], namespace: Dict[str, Any],
                lifetime: Lifetime = Lifetime.TRANSIENT,
                interface: Optional[Type[Any]] = None,
                auto_register: bool = False,
                **kwargs: Any) -> 'ServiceMeta':
        
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        
        # メタ情報を保存
        cls._di_lifetime = lifetime
        cls._di_interface = interface
        cls._di_dependencies: Optional[Dict[str, Type[Any]]] = None
        cls._di_forward_refs_pending = False  # 前方参照が解決できなかったフラグ
        
        # 依存関係を解析してキャッシュ
        if hasattr(cls, '__init__'):
            try:
                hints = get_type_hints(cls.__init__)
                cls._di_dependencies = {
                    k: v for k, v in hints.items() 
                    if k not in ('self', 'return')
                }
            except NameError as e:
                # 前方参照が解決できない場合（クラス未定義）
                # → resolve() 時に再解決を試みる
                cls._di_dependencies = {}
                cls._di_forward_refs_pending = True
            except Exception:
                cls._di_dependencies = {}
        
        # 自動登録
        if auto_register and interface is not None:
            mcs._registry[interface] = ServiceDescriptor(
                service_type=interface,
                implementation_type=cls,
                lifetime=lifetime
            )
        
        return cls
    
    @classmethod
    def get_dependencies(mcs, cls: Type[Any]) -> Dict[str, Type[Any]]:
        """
        クラスの依存関係を取得
        
        前方参照が解決できなかった場合は、ここで再度解決を試みる
        """
        # メタクラス経由でキャッシュされた依存関係があり、
        # かつ前方参照が保留されていない場合はそれを返す
        if (hasattr(cls, '_di_dependencies') and 
            cls._di_dependencies is not None and
            not getattr(cls, '_di_forward_refs_pending', False)):
            return cls._di_dependencies
        
        if not hasattr(cls, '__init__'):
            return {}
        
        try:
            # 実行時に再度 get_type_hints を試みる
            # この時点では全クラスが定義済みのはず
            hints = get_type_hints(cls.__init__)
            deps = {k: v for k, v in hints.items() if k not in ('self', 'return')}
            
            # キャッシュを更新
            if hasattr(cls, '_di_dependencies'):
                cls._di_dependencies = deps
                cls._di_forward_refs_pending = False
            
            return deps
        except Exception:
            return {}


# =============================================================================
# スコープ管理
# =============================================================================

class Scope:
    """
    スコープ付きサービスのライフタイム管理
    コンテキストマネージャとして使用
    """
    
    __slots__ = ('_container', '_instances', '_disposables', '_parent', '_lock', '_id')
    
    _id_counter: ClassVar[int] = 0
    _id_lock: ClassVar[threading.Lock] = threading.Lock()
    
    def __init__(self, container: 'Container', parent: Optional['Scope'] = None):
        self._container = container
        self._instances: Dict[Type[Any], Any] = {}
        self._disposables: List[Union[Disposable, AsyncDisposable]] = []
        self._parent = parent
        self._lock = threading.RLock()
        
        with Scope._id_lock:
            Scope._id_counter += 1
            self._id = Scope._id_counter
    
    @property
    def id(self) -> int:
        return self._id
    
    def get_or_create(self, service_type: Type[T], factory: Callable[[], T]) -> T:
        """スコープ内でインスタンスを取得または作成"""
        with self._lock:
            if service_type in self._instances:
                return cast(T, self._instances[service_type])
            
            instance = factory()
            self._instances[service_type] = instance
            
            # Disposableなら追跡
            if isinstance(instance, (Disposable, AsyncDisposable)):
                self._disposables.append(instance)
            
            return instance
    
    def dispose(self) -> None:
        """同期的にリソースを解放"""
        with self._lock:
            for disposable in reversed(self._disposables):
                if isinstance(disposable, Disposable):
                    try:
                        disposable.dispose()
                    except Exception:
                        pass  # ログに記録すべきだが簡略化
            self._instances.clear()
            self._disposables.clear()
    
    async def dispose_async(self) -> None:
        """非同期的にリソースを解放"""
        with self._lock:
            for disposable in reversed(self._disposables):
                try:
                    if isinstance(disposable, AsyncDisposable):
                        await disposable.dispose_async()
                    elif isinstance(disposable, Disposable):
                        disposable.dispose()
                except Exception:
                    pass
            self._instances.clear()
            self._disposables.clear()


# =============================================================================
# 依存性グラフ解析
# =============================================================================

class DependencyGraph:
    """
    依存関係のグラフ表現と解析
    - 循環依存検出（トポロジカルソート）
    - ライフタイム検証
    - 依存関係の可視化
    """
    
    def __init__(self) -> None:
        self._edges: Dict[Type[Any], Set[Type[Any]]] = defaultdict(set)
        self._lifetimes: Dict[Type[Any], Lifetime] = {}
    
    def add_service(self, service_type: Type[Any], 
                    dependencies: Set[Type[Any]], 
                    lifetime: Lifetime) -> None:
        """サービスとその依存関係を追加"""
        self._edges[service_type] = dependencies
        self._lifetimes[service_type] = lifetime
    
    def detect_cycles(self) -> Optional[List[Type[Any]]]:
        """
        深さ優先探索で循環依存を検出
        循環があれば循環パスを返す
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[Type[Any], int] = defaultdict(int)
        parent: Dict[Type[Any], Optional[Type[Any]]] = {}
        
        def dfs(node: Type[Any]) -> Optional[List[Type[Any]]]:
            color[node] = GRAY
            
            for neighbor in self._edges.get(node, set()):
                if color[neighbor] == GRAY:
                    # 循環検出 - パスを構築
                    cycle = [neighbor, node]
                    current = node
                    while parent.get(current) and parent[current] != neighbor:
                        current = parent[current]
                        cycle.append(current)
                    cycle.append(neighbor)
                    return list(reversed(cycle))
                
                if color[neighbor] == WHITE:
                    parent[neighbor] = node
                    result = dfs(neighbor)
                    if result:
                        return result
            
            color[node] = BLACK
            return None
        
        for node in self._edges:
            if color[node] == WHITE:
                result = dfs(node)
                if result:
                    return result
        
        return None
    
    def validate_lifetimes(self) -> List[Tuple[Type[Any], Type[Any], Lifetime, Lifetime]]:
        """
        ライフタイム不整合を検出
        短命サービス -> 長命サービスへの依存は問題なし
        長命サービス -> 短命サービスへの依存は問題（キャプティブ依存性）
        """
        violations: List[Tuple[Type[Any], Type[Any], Lifetime, Lifetime]] = []
        
        # ライフタイムの順序: SINGLETON > SCOPED > TRANSIENT
        lifetime_order = {
            Lifetime.TRANSIENT: 0,
            Lifetime.SCOPED: 1,
            Lifetime.SINGLETON: 2
        }
        
        for service, deps in self._edges.items():
            service_order = lifetime_order.get(self._lifetimes.get(service, Lifetime.TRANSIENT), 0)
            
            for dep in deps:
                dep_order = lifetime_order.get(self._lifetimes.get(dep, Lifetime.TRANSIENT), 0)
                
                # 長命が短命に依存している場合は警告
                if service_order > dep_order:
                    violations.append((
                        service, dep,
                        self._lifetimes.get(service, Lifetime.TRANSIENT),
                        self._lifetimes.get(dep, Lifetime.TRANSIENT)
                    ))
        
        return violations
    
    def topological_sort(self) -> List[Type[Any]]:
        """トポロジカルソートで初期化順序を決定"""
        in_degree: Dict[Type[Any], int] = defaultdict(int)
        
        for deps in self._edges.values():
            for dep in deps:
                in_degree[dep]  # 存在を保証
        
        for node, deps in self._edges.items():
            in_degree[node]  # 存在を保証
            for dep in deps:
                in_degree[node] += 1
        
        # in_degree が 0 のノードから開始
        queue = [node for node, degree in in_degree.items() if degree == 0]
        result: List[Type[Any]] = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            for dependent, deps in self._edges.items():
                if node in deps:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)
        
        return result


# =============================================================================
# メインコンテナ
# =============================================================================

class Container:
    """
    依存性注入コンテナ
    
    特徴:
    - コンストラクタインジェクション（型ヒントから自動解決）
    - プロパティインジェクション（Injectディスクリプタ）
    - ライフタイム管理（Transient/Singleton/Scoped）
    - 循環依存検出
    - 非同期初期化サポート
    - スレッドセーフ
    
    型安全性について:
        ここでいう「型安全」は、静的型チェッカ（mypy など）で検証可能なレベルの
        安全性を指します。実行時の完全な型保証ではなく、開発時の型エラー検出を
        最大化することが目的です。
    """
    
    def __init__(self) -> None:
        self._descriptors: Dict[Type[Any], ServiceDescriptor] = {}
        self._singletons: Dict[Type[Any], Any] = {}
        # WeakValueDictionary: 参照されなくなったインスタンスを自動解放
        # これによりメモリリークを防止（特に大量のTransientサービスがある場合）
        self._transient_cache: WeakValueDictionary[int, Any] = WeakValueDictionary()
        self._singleton_lock = threading.RLock()
        self._scope_local = threading.local()
        self._graph = DependencyGraph()
        self._validated = False
    
    # -------------------------------------------------------------------------
    # 登録API
    # -------------------------------------------------------------------------
    
    def register(self, descriptor: ServiceDescriptor) -> 'Container':
        """サービス記述子を登録"""
        self._descriptors[descriptor.service_type] = descriptor
        self._validated = False
        return self
    
    def register_singleton(self, service_type: Type[T], 
                          implementation: Optional[Type[T]] = None,
                          instance: Optional[T] = None,
                          factory: Optional[Callable[..., T]] = None) -> 'Container':
        """シングルトンサービスを登録"""
        return self.register(ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation or (service_type if instance is None and factory is None else None),
            instance=instance,
            factory=factory,
            lifetime=Lifetime.SINGLETON
        ))
    
    def register_transient(self, service_type: Type[T],
                          implementation: Optional[Type[T]] = None,
                          factory: Optional[Callable[..., T]] = None) -> 'Container':
        """トランジェントサービスを登録"""
        return self.register(ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation or (service_type if factory is None else None),
            factory=factory,
            lifetime=Lifetime.TRANSIENT
        ))
    
    def register_scoped(self, service_type: Type[T],
                       implementation: Optional[Type[T]] = None,
                       factory: Optional[Callable[..., T]] = None) -> 'Container':
        """スコープ付きサービスを登録"""
        return self.register(ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation or (service_type if factory is None else None),
            factory=factory,
            lifetime=Lifetime.SCOPED
        ))
    
    def register_interface(self, interface: Type[TService], 
                          implementation: Type[TImpl],
                          lifetime: Lifetime = Lifetime.TRANSIENT) -> 'Container':
        """インターフェース→実装のマッピングを登録"""
        return self.register(ServiceDescriptor(
            service_type=interface,
            implementation_type=implementation,
            lifetime=lifetime
        ))
    
    # -------------------------------------------------------------------------
    # 解決API
    # -------------------------------------------------------------------------
    
    def resolve(self, service_type: Type[T]) -> T:
        """サービスを解決（同期）"""
        return self._resolve_internal(service_type, set())
    
    async def resolve_async(self, service_type: Type[T]) -> T:
        """サービスを解決（非同期初期化対応）"""
        instance = self._resolve_internal(service_type, set())
        
        if isinstance(instance, AsyncInitializable):
            await instance.initialize_async()
        
        return instance
    
    def _resolve_internal(self, service_type: Type[T], 
                         resolution_chain: Set[Type[Any]]) -> T:
        """内部解決ロジック"""
        # 循環依存チェック
        if service_type in resolution_chain:
            chain_list = list(resolution_chain) + [service_type]
            raise CircularDependencyError(chain_list)
        
        descriptor = self._descriptors.get(service_type)
        if descriptor is None:
            raise ServiceNotFoundError(service_type)
        
        # 既存インスタンスがあれば返す
        if descriptor.instance is not None:
            return cast(T, descriptor.instance)
        
        # ライフタイムに応じた解決
        if descriptor.lifetime == Lifetime.SINGLETON:
            return self._resolve_singleton(service_type, descriptor, resolution_chain)
        elif descriptor.lifetime == Lifetime.SCOPED:
            return self._resolve_scoped(service_type, descriptor, resolution_chain)
        else:  # TRANSIENT
            return self._create_instance(descriptor, resolution_chain | {service_type})
    
    def _resolve_singleton(self, service_type: Type[T],
                          descriptor: ServiceDescriptor,
                          resolution_chain: Set[Type[Any]]) -> T:
        """シングルトンの解決（ダブルチェックロッキング）"""
        if service_type in self._singletons:
            return cast(T, self._singletons[service_type])
        
        with self._singleton_lock:
            if service_type in self._singletons:
                return cast(T, self._singletons[service_type])
            
            instance = self._create_instance(descriptor, resolution_chain | {service_type})
            self._singletons[service_type] = instance
            return instance
    
    def _resolve_scoped(self, service_type: Type[T],
                       descriptor: ServiceDescriptor,
                       resolution_chain: Set[Type[Any]]) -> T:
        """スコープ付きサービスの解決"""
        scope = self._get_current_scope()
        if scope is None:
            raise ScopeError(
                f"Cannot resolve scoped service {service_type.__name__} "
                "outside of a scope. Use container.scope() context manager."
            )
        
        return scope.get_or_create(
            service_type,
            lambda: self._create_instance(descriptor, resolution_chain | {service_type})
        )
    
    def _create_instance(self, descriptor: ServiceDescriptor,
                        resolution_chain: Set[Type[Any]]) -> Any:
        """インスタンスを生成"""
        # ファクトリがあれば使用
        if descriptor.factory is not None:
            # ファクトリの引数も解決
            sig = inspect.signature(descriptor.factory)
            hints = get_type_hints(descriptor.factory) if hasattr(descriptor.factory, '__annotations__') else {}
            
            kwargs = {}
            for param_name, param in sig.parameters.items():
                if param_name in hints:
                    param_type = hints[param_name]
                    if param_type in self._descriptors:
                        kwargs[param_name] = self._resolve_internal(param_type, resolution_chain)
            
            return descriptor.factory(**kwargs)
        
        # 実装型からインスタンス化
        impl_type = descriptor.implementation_type
        if impl_type is None:
            raise DIException(f"No implementation for {descriptor.service_type.__name__}")
        
        # コンストラクタの依存関係を解決
        dependencies = ServiceMeta.get_dependencies(impl_type)
        kwargs = {}
        
        for param_name, param_type in dependencies.items():
            # Optional型の処理
            origin = get_origin(param_type)
            if origin is Union:
                args = get_args(param_type)
                # Optional[X] は Union[X, None]
                non_none_args = [a for a in args if a is not type(None)]
                if len(non_none_args) == 1:
                    param_type = non_none_args[0]
                    if param_type not in self._descriptors:
                        kwargs[param_name] = None
                        continue
            
            if param_type in self._descriptors:
                kwargs[param_name] = self._resolve_internal(param_type, resolution_chain)
        
        instance = impl_type(**kwargs)
        
        # プロパティインジェクションの処理
        self._inject_properties(instance)
        
        return instance
    
    def _inject_properties(self, instance: Any) -> None:
        """Injectディスクリプタにコンテナをバインド"""
        for name in dir(type(instance)):
            try:
                attr = getattr(type(instance), name)
                if isinstance(attr, Inject):
                    attr.bind_container(self)
            except AttributeError:
                pass
    
    # -------------------------------------------------------------------------
    # スコープ管理
    # -------------------------------------------------------------------------
    
    def _get_current_scope(self) -> Optional[Scope]:
        """現在のスコープを取得"""
        return getattr(self._scope_local, 'current_scope', None)
    
    def _set_current_scope(self, scope: Optional[Scope]) -> None:
        """現在のスコープを設定"""
        self._scope_local.current_scope = scope
    
    @contextmanager
    def scope(self):
        """スコープを作成（同期コンテキストマネージャ）"""
        parent_scope = self._get_current_scope()
        new_scope = Scope(self, parent_scope)
        self._set_current_scope(new_scope)
        
        try:
            yield new_scope
        finally:
            new_scope.dispose()
            self._set_current_scope(parent_scope)
    
    @asynccontextmanager
    async def scope_async(self):
        """スコープを作成（非同期コンテキストマネージャ）"""
        parent_scope = self._get_current_scope()
        new_scope = Scope(self, parent_scope)
        self._set_current_scope(new_scope)
        
        try:
            yield new_scope
        finally:
            await new_scope.dispose_async()
            self._set_current_scope(parent_scope)
    
    # -------------------------------------------------------------------------
    # 検証
    # -------------------------------------------------------------------------
    
    def validate(self, *, strict: bool = False) -> List[str]:
        """
        コンテナの設定を検証
        - 循環依存チェック
        - ライフタイム不整合チェック
        - 解決可能性チェック
        """
        errors: List[str] = []
        
        # グラフを構築
        self._graph = DependencyGraph()
        
        for service_type, descriptor in self._descriptors.items():
            impl = descriptor.implementation_type
            if impl is not None:
                deps = set(ServiceMeta.get_dependencies(impl).values())
                # 登録されている依存のみ
                registered_deps = {d for d in deps if d in self._descriptors}
                self._graph.add_service(service_type, registered_deps, descriptor.lifetime)
        
        # 循環依存チェック
        cycle = self._graph.detect_cycles()
        if cycle:
            errors.append(f"Circular dependency: {' -> '.join(t.__name__ for t in cycle)}")
            if strict:
                raise CircularDependencyError(cycle)
        
        # ライフタイム不整合チェック
        violations = self._graph.validate_lifetimes()
        for consumer, dep, c_life, d_life in violations:
            msg = (f"Lifetime mismatch: {consumer.__name__} ({c_life.name}) "
                   f"depends on {dep.__name__} ({d_life.name})")
            errors.append(msg)
            if strict:
                raise LifetimeMismatchError(consumer, dep, c_life, d_life)
        
        self._validated = True
        return errors
    
    def build(self) -> 'Container':
        """検証してコンテナを確定"""
        self.validate(strict=True)
        return self


# =============================================================================
# デコレータAPI
# =============================================================================

def injectable(lifetime: Lifetime = Lifetime.TRANSIENT,
               interface: Optional[Type[Any]] = None):
    """
    クラスを注入可能としてマークするデコレータ
    
    使用例:
        @injectable(lifetime=Lifetime.SINGLETON)
        class MyService:
            pass
    """
    def decorator(cls: Type[T]) -> Type[T]:
        cls._di_lifetime = lifetime
        cls._di_interface = interface
        return cls
    return decorator

def singleton(cls: Type[T]) -> Type[T]:
    """シングルトンとしてマーク"""
    cls._di_lifetime = Lifetime.SINGLETON
    return cls

def scoped(cls: Type[T]) -> Type[T]:
    """スコープ付きとしてマーク"""
    cls._di_lifetime = Lifetime.SCOPED
    return cls

def transient(cls: Type[T]) -> Type[T]:
    """トランジェントとしてマーク"""
    cls._di_lifetime = Lifetime.TRANSIENT
    return cls


def inject(*service_types: Type[Any]):
    """
    メソッドに依存を注入するデコレータ
    
    使用例:
        @inject(ILogger, IConfig)
        def process(self, logger: ILogger, config: IConfig):
            pass
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # 実行時にコンテナを見つけて注入
            # 注: 実際の実装ではコンテナ参照が必要
            return func(*args, **kwargs)
        wrapper._inject_types = service_types
        return wrapper
    return decorator


# =============================================================================
# ビルダーパターン
# =============================================================================

class ContainerBuilder:
    """
    流暢なAPIでコンテナを構築
    
    使用例:
        container = (ContainerBuilder()
            .add_singleton(ILogger, ConsoleLogger)
            .add_transient(IService, ServiceImpl)
            .add_scoped(IDbContext, DbContext)
            .build())
    """
    
    def __init__(self) -> None:
        self._container = Container()
    
    def add_singleton(self, service_type: Type[T],
                     implementation: Optional[Type[T]] = None,
                     instance: Optional[T] = None,
                     factory: Optional[Callable[..., T]] = None) -> 'ContainerBuilder':
        self._container.register_singleton(service_type, implementation, instance, factory)
        return self
    
    def add_transient(self, service_type: Type[T],
                     implementation: Optional[Type[T]] = None,
                     factory: Optional[Callable[..., T]] = None) -> 'ContainerBuilder':
        self._container.register_transient(service_type, implementation, factory)
        return self
    
    def add_scoped(self, service_type: Type[T],
                  implementation: Optional[Type[T]] = None,
                  factory: Optional[Callable[..., T]] = None) -> 'ContainerBuilder':
        self._container.register_scoped(service_type, implementation, factory)
        return self
    
    def add_interface(self, interface: Type[TService],
                     implementation: Type[TImpl],
                     lifetime: Lifetime = Lifetime.TRANSIENT) -> 'ContainerBuilder':
        self._container.register_interface(interface, implementation, lifetime)
        return self
    
    def build(self) -> Container:
        return self._container.build()


# =============================================================================
# 使用例とデモ
# =============================================================================

if __name__ == "__main__":
    from abc import ABC, abstractmethod
    import asyncio
    
    # --- インターフェース定義 ---
    
    class ILogger(ABC):
        @abstractmethod
        def log(self, message: str) -> None: ...
    
    class IUserRepository(ABC):
        @abstractmethod
        def get_user(self, user_id: int) -> dict: ...
    
    class IEmailService(ABC):
        @abstractmethod
        async def send_email(self, to: str, subject: str, body: str) -> bool: ...
    
    class IUserService(ABC):
        @abstractmethod
        def get_user_info(self, user_id: int) -> str: ...
    
    # --- 実装クラス ---
    
    @singleton
    class ConsoleLogger(ILogger):
        def __init__(self) -> None:
            print("ConsoleLogger: 初期化されました")
        
        def log(self, message: str) -> None:
            print(f"[LOG] {message}")
        
        def dispose(self) -> None:
            print("ConsoleLogger: 破棄されました")
    
    @scoped
    class UserRepository(IUserRepository):
        def __init__(self, logger: ILogger) -> None:
            self._logger = logger
            self._logger.log("UserRepository: 初期化されました")
        
        def get_user(self, user_id: int) -> dict:
            self._logger.log(f"ユーザー取得: ID={user_id}")
            return {"id": user_id, "name": f"User_{user_id}", "email": f"user{user_id}@example.com"}
        
        def dispose(self) -> None:
            self._logger.log("UserRepository: 破棄されました")
    
    class EmailService(IEmailService):
        def __init__(self, logger: ILogger) -> None:
            self._logger = logger
            self._initialized = False
        
        async def initialize_async(self) -> None:
            self._logger.log("EmailService: 非同期初期化中...")
            await asyncio.sleep(0.1)  # SMTP接続シミュレーション
            self._initialized = True
            self._logger.log("EmailService: 初期化完了")
        
        async def send_email(self, to: str, subject: str, body: str) -> bool:
            if not self._initialized:
                raise RuntimeError("EmailService が初期化されていません")
            self._logger.log(f"メール送信: To={to}, Subject={subject}")
            await asyncio.sleep(0.05)  # 送信シミュレーション
            return True
        
        async def dispose_async(self) -> None:
            self._logger.log("EmailService: 非同期破棄")
    
    @transient
    class UserService(IUserService):
        # プロパティインジェクションのデモ
        email_service: Inject[IEmailService]
        
        def __init__(self, logger: ILogger, user_repo: IUserRepository) -> None:
            self._logger = logger
            self._user_repo = user_repo
        
        def get_user_info(self, user_id: int) -> str:
            user = self._user_repo.get_user(user_id)
            return f"ID: {user['id']}, 名前: {user['name']}, Email: {user['email']}"
    
    # --- コンテナ構築と使用 ---
    
    print("=" * 60)
    print("PyDI デモ: 依存性注入コンテナ")
    print("=" * 60)
    
    # ビルダーパターンでコンテナ構築
    container = (ContainerBuilder()
        .add_singleton(ILogger, ConsoleLogger)
        .add_scoped(IUserRepository, UserRepository)
        .add_transient(IEmailService, EmailService)
        .add_transient(IUserService, UserService)
        .build())
    
    print("\n--- 検証完了 ---\n")
    
    # スコープを使用したサービス解決
    print("\n--- スコープ1 ---")
    with container.scope() as scope1:
        user_service1 = container.resolve(IUserService)
        print(user_service1.get_user_info(1))
        
        # 同じスコープ内では同じUserRepositoryインスタンス
        user_service2 = container.resolve(IUserService)
        print(user_service2.get_user_info(2))
    
    print("\n--- スコープ2 (新しいUserRepository) ---")
    with container.scope() as scope2:
        user_service3 = container.resolve(IUserService)
        print(user_service3.get_user_info(3))
    
    # 非同期デモ
    print("\n--- 非同期サービスデモ ---")
    
    async def async_demo():
        async with container.scope_async() as scope:
            # 非同期初期化
            email_svc = await container.resolve_async(IEmailService)
            await email_svc.send_email("test@example.com", "テスト", "Hello!")
    
    asyncio.run(async_demo())
    
    print("\n" + "=" * 60)
    print("デモ完了")
    print("=" * 60)
