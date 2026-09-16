from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _tree(relative_path: str) -> ast.Module:
    return ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))


def _find_function(relative_path: str, function_name: str) -> ast.FunctionDef:
    for node in ast.walk(_tree(relative_path)):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    raise AssertionError(f"{function_name} not found in {relative_path}")


def _defaults(function: ast.FunctionDef) -> dict[str, object]:
    positional = function.args.posonlyargs + function.args.args
    names = [argument.arg for argument in positional[-len(function.args.defaults) :]]
    values = [ast.literal_eval(value) for value in function.args.defaults]
    keyword_values = {
        argument.arg: ast.literal_eval(value)
        for argument, value in zip(function.args.kwonlyargs, function.args.kw_defaults)
        if value is not None
    }
    return dict(zip(names, values)) | keyword_values


def _annotated_default(
    relative_path: str, class_name: str, attribute_name: str
) -> object:
    for node in ast.walk(_tree(relative_path)):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for statement in node.body:
                if (
                    isinstance(statement, ast.AnnAssign)
                    and isinstance(statement.target, ast.Name)
                    and statement.target.id == attribute_name
                ):
                    return ast.literal_eval(statement.value)
    raise AssertionError(
        f"{class_name}.{attribute_name} not found in {relative_path}"
    )


class GenerationDefaultsTest(unittest.TestCase):
    def test_llada_model_defaults(self):
        llada20 = _defaults(
            _find_function("src/model/llada2_0/modeling.py", "generate")
        )
        llada21 = _defaults(
            _find_function("src/model/llada2_1/modeling.py", "generate")
        )

        self.assertEqual(llada20["threshold"], 0.95)
        self.assertEqual(llada21["threshold"], 0.7)
        self.assertEqual(llada21["editing_threshold"], 0.5)

    def test_focus_defaults(self):
        focus = _defaults(
            _find_function("src/focus/generation.py", "focus_generate")
        )
        focus_v2 = _defaults(
            _find_function("src/focus_v2/generation.py", "focus_v2_generate")
        )

        self.assertEqual(focus["editing_threshold"], 0.5)
        self.assertEqual(focus_v2["editing_threshold"], 0.5)
        for relative_path in (
            "src/focus/generation.py",
            "src/focus_v2/generation.py",
        ):
            source = (ROOT / relative_path).read_text(encoding="utf-8")
            self.assertIn(
                'threshold = 0.7 if family == "llada" else 0.95', source
            )

    def test_losa_defaults(self):
        for relative_path in (
            "src/losa/generation.py",
            "src/losa_v2/generation.py",
        ):
            defaults = _defaults(
                _find_function(relative_path, "block_diffusion_generate")
            )
            self.assertEqual(defaults["editing_threshold"], 0.5)
            self.assertEqual(defaults["losa_token_budget"], 256)

        self.assertEqual(
            _annotated_default(
                "src/losa/api.py", "LoSARuntime", "losa_token_budget"
            ),
            256,
        )
        self.assertEqual(
            _annotated_default(
                "src/losa_v2/api.py", "LoSAV2Runtime", "losa_token_budget"
            ),
            256,
        )
        for relative_path in (
            "src/losa/generation.py",
            "src/losa_v2/generation.py",
        ):
            source = (ROOT / relative_path).read_text(encoding="utf-8")
            self.assertIn(
                'remasking_strategy == "low_confidence_dynamic"', source
            )
            self.assertIn("else 0.95", source)
            self.assertIn("else 0.85", source)

    def test_sparse_direct_patch_defaults_match_public_config(self):
        llada_patch = _defaults(
            _find_function("src/sparse/llada_patch.py", "patch_llada_model")
        )
        sdar_selector = _defaults(
            _find_function("src/sparse/sdar_patch.py", "_select_positions")
        )

        self.assertEqual(llada_patch["ratio"], 0.7)
        self.assertEqual(sdar_selector["threshold"], 0.85)


if __name__ == "__main__":
    unittest.main()
