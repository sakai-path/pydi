"""
PyDI - 依存性注入コンテナ インタラクティブデモ
Streamlitで動作確認できるデモアプリケーション
"""

import streamlit as st
import sys
from io import StringIO
from contextlib import redirect_stdout
from abc import ABC, abstractmethod
from typing import Optional
import time

# ページ設定
st.set_page_config(
    page_title="PyDI - Python DI Container Demo",
    page_icon="💉",
    layout="wide"
)

# カスタムCSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 20px; }
    .code-output { 
        background-color: #1e1e1e; 
        color: #d4d4d4; 
        padding: 15px; 
        border-radius: 5px;
        font-family: 'Consolas', monospace;
    }
    .success-box { 
        background-color: #d4edda; 
        border: 1px solid #c3e6cb; 
        padding: 10px; 
        border-radius: 5px; 
    }
    .error-box { 
        background-color: #f8d7da; 
        border: 1px solid #f5c6cb; 
        padding: 10px; 
        border-radius: 5px; 
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 簡易版DIコンテナ（デモ用）
# =============================================================================

from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, Callable, Dict, Type, TypeVar, Set, List
from collections import defaultdict
import threading

T = TypeVar('T')

class Lifetime(Enum):
    TRANSIENT = auto()
    SINGLETON = auto()
    SCOPED = auto()

@dataclass
class ServiceDescriptor:
    service_type: Type[Any]
    implementation_type: Type[Any]
    lifetime: Lifetime = Lifetime.TRANSIENT

class CircularDependencyError(Exception):
    def __init__(self, chain: List[Type[Any]]):
        self.chain = chain
        chain_str = " → ".join(t.__name__ for t in chain)
        super().__init__(f"循環依存を検出: {chain_str}")

class LifetimeMismatchWarning:
    def __init__(self, consumer: Type, dependency: Type, c_life: Lifetime, d_life: Lifetime):
        self.message = f"⚠️ ライフタイム不整合: {consumer.__name__} ({c_life.name}) → {dependency.__name__} ({d_life.name})"

class DependencyGraph:
    def __init__(self):
        self._edges: Dict[Type, Set[Type]] = defaultdict(set)
        self._lifetimes: Dict[Type, Lifetime] = {}
    
    def add_service(self, service_type: Type, deps: Set[Type], lifetime: Lifetime):
        self._edges[service_type] = deps
        self._lifetimes[service_type] = lifetime
    
    def detect_cycles(self) -> Optional[List[Type]]:
        WHITE, GRAY, BLACK = 0, 1, 2
        color = defaultdict(int)
        parent = {}
        
        def dfs(node):
            color[node] = GRAY
            for neighbor in self._edges.get(node, set()):
                if color[neighbor] == GRAY:
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
    
    def check_lifetime_issues(self) -> List[LifetimeMismatchWarning]:
        warnings = []
        lifetime_order = {Lifetime.TRANSIENT: 0, Lifetime.SCOPED: 1, Lifetime.SINGLETON: 2}
        
        for service, deps in self._edges.items():
            s_order = lifetime_order.get(self._lifetimes.get(service, Lifetime.TRANSIENT), 0)
            for dep in deps:
                d_order = lifetime_order.get(self._lifetimes.get(dep, Lifetime.TRANSIENT), 0)
                if s_order > d_order:
                    warnings.append(LifetimeMismatchWarning(
                        service, dep,
                        self._lifetimes.get(service, Lifetime.TRANSIENT),
                        self._lifetimes.get(dep, Lifetime.TRANSIENT)
                    ))
        return warnings
    
    def to_mermaid(self) -> str:
        """Mermaid形式でグラフを出力"""
        lines = ["graph TD"]
        lifetime_styles = {
            Lifetime.SINGLETON: "fill:#e1f5fe,stroke:#01579b",
            Lifetime.SCOPED: "fill:#fff3e0,stroke:#e65100",
            Lifetime.TRANSIENT: "fill:#f3e5f5,stroke:#7b1fa2"
        }
        
        node_ids = {}
        for i, node in enumerate(self._edges.keys()):
            node_ids[node] = f"N{i}"
            lifetime = self._lifetimes.get(node, Lifetime.TRANSIENT)
            label = f"{node.__name__}<br/>({lifetime.name})"
            lines.append(f'    {node_ids[node]}["{label}"]')
        
        for service, deps in self._edges.items():
            for dep in deps:
                if dep in node_ids:
                    lines.append(f"    {node_ids[service]} --> {node_ids[dep]}")
        
        # スタイル適用
        for node, node_id in node_ids.items():
            lifetime = self._lifetimes.get(node, Lifetime.TRANSIENT)
            style = lifetime_styles.get(lifetime, "")
            lines.append(f"    style {node_id} {style}")
        
        return "\n".join(lines)


class MiniContainer:
    """デモ用の簡易DIコンテナ"""
    
    def __init__(self):
        self._descriptors: Dict[Type, ServiceDescriptor] = {}
        self._singletons: Dict[Type, Any] = {}
        self._scoped: Dict[Type, Any] = {}
        self._in_scope = False
        self._resolution_log: List[str] = []
    
    def register(self, service_type: Type, impl_type: Type = None, lifetime: Lifetime = Lifetime.TRANSIENT):
        self._descriptors[service_type] = ServiceDescriptor(
            service_type=service_type,
            implementation_type=impl_type or service_type,
            lifetime=lifetime
        )
        return self
    
    def build_graph(self) -> DependencyGraph:
        graph = DependencyGraph()
        for service_type, desc in self._descriptors.items():
            impl = desc.implementation_type
            deps = set()
            if hasattr(impl, '__init__'):
                import inspect
                sig = inspect.signature(impl.__init__)
                for param_name, param in sig.parameters.items():
                    if param_name != 'self' and param.annotation != inspect.Parameter.empty:
                        if param.annotation in self._descriptors:
                            deps.add(param.annotation)
            graph.add_service(service_type, deps, desc.lifetime)
        return graph
    
    def validate(self) -> tuple[bool, List[str]]:
        graph = self.build_graph()
        errors = []
        
        cycle = graph.detect_cycles()
        if cycle:
            errors.append(f"❌ 循環依存: {' → '.join(t.__name__ for t in cycle)}")
        
        warnings = graph.check_lifetime_issues()
        for w in warnings:
            errors.append(w.message)
        
        return len([e for e in errors if e.startswith("❌")]) == 0, errors
    
    def enter_scope(self):
        self._in_scope = True
        self._scoped.clear()
    
    def exit_scope(self):
        self._in_scope = False
        self._scoped.clear()
    
    def resolve(self, service_type: Type[T], chain: Set[Type] = None) -> T:
        chain = chain or set()
        
        if service_type in chain:
            raise CircularDependencyError(list(chain) + [service_type])
        
        desc = self._descriptors.get(service_type)
        if not desc:
            raise ValueError(f"未登録: {service_type.__name__}")
        
        # ライフタイムに応じた解決
        if desc.lifetime == Lifetime.SINGLETON:
            if service_type not in self._singletons:
                self._resolution_log.append(f"🔵 SINGLETON 新規作成: {service_type.__name__}")
                self._singletons[service_type] = self._create(desc, chain | {service_type})
            else:
                self._resolution_log.append(f"🔵 SINGLETON キャッシュ: {service_type.__name__}")
            return self._singletons[service_type]
        
        elif desc.lifetime == Lifetime.SCOPED:
            if not self._in_scope:
                raise ValueError(f"スコープ外でSCOPEDサービスを解決できません: {service_type.__name__}")
            if service_type not in self._scoped:
                self._resolution_log.append(f"🟠 SCOPED 新規作成: {service_type.__name__}")
                self._scoped[service_type] = self._create(desc, chain | {service_type})
            else:
                self._resolution_log.append(f"🟠 SCOPED キャッシュ: {service_type.__name__}")
            return self._scoped[service_type]
        
        else:  # TRANSIENT
            self._resolution_log.append(f"🟣 TRANSIENT 新規作成: {service_type.__name__}")
            return self._create(desc, chain | {service_type})
    
    def _create(self, desc: ServiceDescriptor, chain: Set[Type]) -> Any:
        impl = desc.implementation_type
        import inspect
        sig = inspect.signature(impl.__init__)
        
        kwargs = {}
        for param_name, param in sig.parameters.items():
            if param_name != 'self' and param.annotation != inspect.Parameter.empty:
                if param.annotation in self._descriptors:
                    kwargs[param_name] = self.resolve(param.annotation, chain)
        
        return impl(**kwargs)
    
    def get_resolution_log(self) -> List[str]:
        return self._resolution_log
    
    def clear_log(self):
        self._resolution_log.clear()


# =============================================================================
# Streamlit UI
# =============================================================================

st.title("💉 PyDI - Python 依存性注入コンテナ デモ")
st.markdown("メタプログラミングを駆使した型安全なDIコンテナの動作を確認できます")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 依存グラフ可視化", 
    "🔄 循環依存検出", 
    "⏰ ライフタイム比較",
    "🎮 インタラクティブデモ"
])

# =============================================================================
# Tab 1: 依存グラフ可視化
# =============================================================================
with tab1:
    st.header("依存関係グラフの可視化")
    
    st.markdown("""
    DIコンテナは内部で依存関係をグラフとして管理しています。
    以下は典型的なWebアプリケーションの依存構造です。
    """)
    
    # サンプルクラス定義
    class ILogger(ABC):
        @abstractmethod
        def log(self, msg: str): ...
    
    class ICache(ABC):
        @abstractmethod
        def get(self, key: str): ...
    
    class IUserRepository(ABC):
        @abstractmethod
        def find(self, id: int): ...
    
    class IUserService(ABC):
        @abstractmethod
        def get_user(self, id: int): ...
    
    class ConsoleLogger(ILogger):
        def log(self, msg: str):
            print(f"[LOG] {msg}")
    
    class RedisCache(ICache):
        def __init__(self, logger: ILogger):
            self.logger = logger
        def get(self, key: str):
            return None
    
    class UserRepository(IUserRepository):
        def __init__(self, logger: ILogger, cache: ICache):
            self.logger = logger
            self.cache = cache
        def find(self, id: int):
            return {"id": id}
    
    class UserService(IUserService):
        def __init__(self, repo: IUserRepository, logger: ILogger):
            self.repo = repo
            self.logger = logger
        def get_user(self, id: int):
            return self.repo.find(id)
    
    # コンテナ構築
    container = MiniContainer()
    container.register(ILogger, ConsoleLogger, Lifetime.SINGLETON)
    container.register(ICache, RedisCache, Lifetime.SINGLETON)
    container.register(IUserRepository, UserRepository, Lifetime.SCOPED)
    container.register(IUserService, UserService, Lifetime.TRANSIENT)
    
    graph = container.build_graph()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("依存関係グラフ")
        mermaid_code = graph.to_mermaid()
        st.code(mermaid_code, language="mermaid")
        
        # Mermaidレンダリング
        st.markdown(f"""
```mermaid
{mermaid_code}
```
        """)
    
    with col2:
        st.subheader("凡例")
        st.markdown("""
        | 色 | ライフタイム | 説明 |
        |:--:|:--|:--|
        | 🔵 | SINGLETON | アプリ全体で1つ |
        | 🟠 | SCOPED | スコープ内で1つ |
        | 🟣 | TRANSIENT | 毎回新規作成 |
        """)
        
        st.subheader("登録されたサービス")
        for svc, desc in container._descriptors.items():
            st.markdown(f"- `{svc.__name__}` → `{desc.implementation_type.__name__}` ({desc.lifetime.name})")

# =============================================================================
# Tab 2: 循環依存検出
# =============================================================================
with tab2:
    st.header("🔄 循環依存の検出")
    
    st.markdown("""
    循環依存とは、A → B → C → A のように依存がループする状態です。
    DIコンテナはこれを検出してエラーを出します。
    """)
    
    # 循環依存のあるクラス
    class ServiceA:
        def __init__(self, b: 'ServiceB'):
            self.b = b
    
    class ServiceB:
        def __init__(self, c: 'ServiceC'):
            self.c = c
    
    class ServiceC:
        def __init__(self, a: 'ServiceA'):
            self.a = a
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("❌ 循環依存あり")
        st.code("""
class ServiceA:
    def __init__(self, b: ServiceB): ...

class ServiceB:
    def __init__(self, c: ServiceC): ...

class ServiceC:
    def __init__(self, a: ServiceA): ...  # ← 循環！
        """, language="python")
        
        if st.button("循環依存を検出", key="detect_cycle"):
            container_bad = MiniContainer()
            container_bad.register(ServiceA)
            container_bad.register(ServiceB)
            container_bad.register(ServiceC)
            
            graph = container_bad.build_graph()
            cycle = graph.detect_cycles()
            
            if cycle:
                st.error(f"🚨 循環依存を検出しました！")
                st.markdown(f"**検出されたサイクル:** {' → '.join(t.__name__ for t in cycle)}")
                
                # 循環のMermaid表示
                st.markdown(f"""
```mermaid
graph LR
    ServiceA --> ServiceB
    ServiceB --> ServiceC
    ServiceC -->|循環!| ServiceA
    style ServiceA fill:#ffcdd2
    style ServiceB fill:#ffcdd2
    style ServiceC fill:#ffcdd2
```
                """)
    
    with col2:
        st.subheader("✅ 循環依存なし")
        st.code("""
class Logger:
    pass

class Repository:
    def __init__(self, logger: Logger): ...

class Service:
    def __init__(self, repo: Repository): ...
        """, language="python")
        
        class Logger2:
            pass
        
        class Repository2:
            def __init__(self, logger: Logger2):
                self.logger = logger
        
        class Service2:
            def __init__(self, repo: Repository2):
                self.repo = repo
        
        if st.button("検証する", key="validate_good"):
            container_good = MiniContainer()
            container_good.register(Logger2)
            container_good.register(Repository2)
            container_good.register(Service2)
            
            graph = container_good.build_graph()
            cycle = graph.detect_cycles()
            
            if cycle is None:
                st.success("✅ 循環依存はありません！")
                st.markdown(f"""
```mermaid
graph TD
    Service2 --> Repository2
    Repository2 --> Logger2
    style Service2 fill:#c8e6c9
    style Repository2 fill:#c8e6c9
    style Logger2 fill:#c8e6c9
```
                """)

# =============================================================================
# Tab 3: ライフタイム比較
# =============================================================================
with tab3:
    st.header("⏰ ライフタイムの違いを体験")
    
    st.markdown("""
    3種類のライフタイムの違いを実際に確認してみましょう。
    """)
    
    # カウンター付きクラス
    class CountedService:
        _counter = 0
        
        def __init__(self):
            CountedService._counter += 1
            self.id = CountedService._counter
        
        @classmethod
        def reset(cls):
            cls._counter = 0
    
    class SingletonService(CountedService):
        _counter = 0
        def __init__(self):
            SingletonService._counter += 1
            self.id = SingletonService._counter
        @classmethod
        def reset(cls):
            cls._counter = 0
    
    class ScopedService(CountedService):
        _counter = 0
        def __init__(self):
            ScopedService._counter += 1
            self.id = ScopedService._counter
        @classmethod
        def reset(cls):
            cls._counter = 0
    
    class TransientService(CountedService):
        _counter = 0
        def __init__(self):
            TransientService._counter += 1
            self.id = TransientService._counter
        @classmethod
        def reset(cls):
            cls._counter = 0
    
    if st.button("🔄 リセット & テスト実行"):
        SingletonService.reset()
        ScopedService.reset()
        TransientService.reset()
        
        container = MiniContainer()
        container.register(SingletonService, lifetime=Lifetime.SINGLETON)
        container.register(ScopedService, lifetime=Lifetime.SCOPED)
        container.register(TransientService, lifetime=Lifetime.TRANSIENT)
        
        results = []
        
        # スコープ1
        container.enter_scope()
        results.append("=== スコープ 1 ===")
        for i in range(3):
            s = container.resolve(SingletonService)
            sc = container.resolve(ScopedService)
            t = container.resolve(TransientService)
            results.append(f"  解決 {i+1}: Singleton=#{s.id}, Scoped=#{sc.id}, Transient=#{t.id}")
        container.exit_scope()
        
        # スコープ2
        container.enter_scope()
        results.append("\n=== スコープ 2 ===")
        for i in range(3):
            s = container.resolve(SingletonService)
            sc = container.resolve(ScopedService)
            t = container.resolve(TransientService)
            results.append(f"  解決 {i+1}: Singleton=#{s.id}, Scoped=#{sc.id}, Transient=#{t.id}")
        container.exit_scope()
        
        st.code("\n".join(results))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔵 SINGLETON", "常に #1", "アプリ全体で共有")
        with col2:
            st.metric("🟠 SCOPED", "#1 → #2", "スコープごとに新規")
        with col3:
            st.metric("🟣 TRANSIENT", "#1〜#6", "毎回新規作成")

# =============================================================================
# Tab 4: インタラクティブデモ
# =============================================================================
with tab4:
    st.header("🎮 カスタム依存関係を試す")
    
    st.markdown("自分で依存関係を定義して、DIコンテナの動作を確認できます。")
    
    # セッション状態の初期化
    if 'custom_services' not in st.session_state:
        st.session_state.custom_services = []
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("サービスを追加")
        
        service_name = st.text_input("サービス名", "MyService")
        depends_on = st.multiselect(
            "依存先", 
            [s['name'] for s in st.session_state.custom_services],
            help="このサービスが依存する他のサービスを選択"
        )
        lifetime = st.selectbox("ライフタイム", ["TRANSIENT", "SCOPED", "SINGLETON"])
        
        if st.button("➕ サービスを追加"):
            st.session_state.custom_services.append({
                'name': service_name,
                'depends_on': depends_on,
                'lifetime': lifetime
            })
            st.rerun()
        
        if st.button("🗑️ 全てクリア"):
            st.session_state.custom_services = []
            st.rerun()
    
    with col2:
        st.subheader("登録済みサービス")
        
        if st.session_state.custom_services:
            for i, svc in enumerate(st.session_state.custom_services):
                deps = " → ".join(svc['depends_on']) if svc['depends_on'] else "(なし)"
                st.markdown(f"**{i+1}. {svc['name']}** ({svc['lifetime']})")
                st.markdown(f"   依存: {deps}")
        else:
            st.info("サービスを追加してください")
    
    if st.session_state.custom_services and st.button("🔍 依存関係を分析"):
        # 動的にクラスを生成
        service_classes = {}
        
        for svc in st.session_state.custom_services:
            # 依存の型ヒントを持つクラスを動的に作成
            deps = {d: service_classes[d] for d in svc['depends_on'] if d in service_classes}
            
            # __init__ を動的に作成
            def make_init(deps_dict):
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)
                # アノテーションを追加
                __init__.__annotations__ = deps_dict
                return __init__
            
            new_class = type(svc['name'], (), {'__init__': make_init(deps)})
            service_classes[svc['name']] = new_class
        
        # コンテナに登録
        container = MiniContainer()
        for svc in st.session_state.custom_services:
            lt = getattr(Lifetime, svc['lifetime'])
            container.register(service_classes[svc['name']], lifetime=lt)
        
        # 検証
        graph = container.build_graph()
        
        # 循環チェック
        cycle = graph.detect_cycles()
        if cycle:
            st.error(f"🚨 循環依存を検出: {' → '.join(t.__name__ for t in cycle)}")
        else:
            st.success("✅ 循環依存なし")
        
        # ライフタイム警告
        warnings = graph.check_lifetime_issues()
        if warnings:
            st.warning("⚠️ ライフタイム不整合の警告:")
            for w in warnings:
                st.markdown(f"- {w.message}")
        
        # グラフ表示
        st.subheader("依存関係グラフ")
        mermaid = graph.to_mermaid()
        st.code(mermaid, language="mermaid")

# フッター
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <p>PyDI - Python Dependency Injection Container</p>
    <p>Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)
