"""Guard canonical source paths, runtime identities and local import targets."""
import ast
import importlib
import importlib.util
from pathlib import Path

from src.runtime import load_runtime, method_source_directory

ROOT = Path(__file__).resolve().parents[1]


def test_explicit_source_families():
    for name in ("sparse", "losa", "focus"):
        assert (ROOT / "src" / "reference" / name / "__init__.py").is_file()
        assert (ROOT / "src" / "optimized" / name / "__init__.py").is_file()
    assert (ROOT / "src/optimized/dense/api.py").is_file()
    assert not list((ROOT / "src").glob("*_v[0-9]*"))
    assert not (ROOT / "src/kernels/versioned").exists()


def test_runtime_identities():
    for family in ("llada", "sdar"):
        for method in ("dense_optimized", "losa_optimized", "focus_optimized"):
            runtime = load_runtime(method, family=family)
            assert type(runtime).__module__.startswith("src.optimized.")
            assert (ROOT / method_source_directory(method) / "api.py").is_file()
        for method in ("losa", "focus"):
            assert type(load_runtime(method, family=family)).__module__.startswith("src.reference.")


def test_src_imports_resolve_inside_checkout():
    modules = set()
    for top in ("src", "tests", "scripts", "eval_instruct"):
        for file in (ROOT / top).rglob("*.py"):
            if file.name.startswith("._"):
                continue
            for node in ast.walk(ast.parse(file.read_text(), filename=str(file))):
                if isinstance(node, ast.ImportFrom) and not node.level and node.module:
                    modules.add(node.module)
                if isinstance(node, ast.Import):
                    modules.update(a.name for a in node.names)
    for module in sorted(m for m in modules if m.startswith("src.")):
        spec = importlib.util.find_spec(module)
        assert spec is not None, module
        if spec.origin:
            assert Path(spec.origin).resolve().is_relative_to(ROOT), module
