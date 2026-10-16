# MathQA 200-question pool

`test_200.jsonl` is a deterministic 200-question sample drawn without
replacement from the official MathQA **test** split. It is divided into 100
`development` and 100 `reserved_test` entries for the experiment protocol; both
partitions still originate from MathQA's official test split.

Official sources:

- Homepage: <https://math-qa.github.io/math-QA/>
- Archive: <https://math-qa.github.io/math-QA/data/MathQA.zip>
- Paper: <https://arxiv.org/abs/1905.13319>

`manifest.json` records the source archive/test hashes, all selected original
indices, the seed, and the output hash. Rebuild it with:

```powershell
$env:PYTHONPATH = (Resolve-Path "src")
python scripts/prepare_mathqa.py
```

The cached official archive is stored under `data/raw/` and is intentionally
ignored by Git. Runtime `Question` objects contain only the problem and choices;
the answer key remains in the benchmark record for independent grading.

`pilot_20.jsonl` is the smaller, fixed pipeline test: the first 10 records from
each partition in parent-file order. Its separate manifest records the parent
and output hashes. Rebuild it without downloading data:

```powershell
python scripts/prepare_pilot20.py
```
