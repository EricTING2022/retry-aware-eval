from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from retry_eval.benchmark import load_mathqa_benchmark, parse_mathqa_options


class MathQABenchmarkTests(unittest.TestCase):
    def test_options_are_parsed_without_splitting_numeric_commas(self) -> None:
        options = parse_mathqa_options(
            "a ) 1,000 , b ) 2,000 , c ) 3,000 , d ) 4,000 , e ) 5,000"
        )
        self.assertEqual(options["a"], "1,000")
        self.assertEqual(options["e"], "5,000")

    def test_stringified_official_option_lists_are_supported(self) -> None:
        options = parse_mathqa_options(
            "['a ) 1 feet', 'b ) 2 feet', 'c ) 3 feet', 'd ) 4 feet', 'e ) 5 feet']"
        )
        self.assertEqual(options["c"], "3 feet")

    def test_loader_keeps_answer_key_out_of_runtime_question(self) -> None:
        row = {
            "question_id": "mathqa-test-0001",
            "source_index": 1,
            "partition": "development",
            "problem": "What is 1+1?",
            "options": {"a": "1", "b": "2", "c": "3", "d": "4", "e": "5"},
            "correct_option": "b",
            "category": "general",
        }
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "benchmark.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            item = load_mathqa_benchmark(path)[0]
        question = item.question()
        self.assertEqual(item.correct_option, "b")
        self.assertFalse(hasattr(question, "correct_option"))
        self.assertEqual(question.options, "a) 1, b) 2, c) 3, d) 4, e) 5")

    def test_checked_in_official_pool_matches_its_manifest(self) -> None:
        project = Path(__file__).resolve().parents[1]
        benchmark_path = project / "data" / "mathqa" / "test_200.jsonl"
        manifest = json.loads(
            (project / "data" / "mathqa" / "manifest.json").read_text(encoding="utf-8")
        )
        items = load_mathqa_benchmark(benchmark_path)
        self.assertEqual(len(items), 200)
        self.assertEqual(
            Counter(item.partition for item in items),
            {"development": 100, "reserved_test": 100},
        )
        self.assertEqual(
            hashlib.sha256(benchmark_path.read_bytes()).hexdigest(),
            manifest["output_sha256"],
        )
        self.assertEqual(manifest["source_test_count"], 2985)


if __name__ == "__main__":
    unittest.main()
