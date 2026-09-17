"""The evaluation population, frozen to a file.

    python -m baselines.common.cases --config lvfm/configs/mrflow_STDiT-L2_16f8.yaml \
        --split test --n_per_bucket 100 --out baselines/cases-test-n100.json

**Why freeze it at all.** `evaluation/main.py` derives its case list at runtime, from `list_series`
at a fixed seed plus `select_cases`. That is reproducible for one model run today, but a baseline
table is built over weeks: MRFlow generated in March, GenerateCT in April, Text2CT in May. A
changed `series.parquet`, a changed `mri.max_series_test` or a changed `list_series` would move the
population underneath the table with nothing in any output to say so -- every row would still be
labelled "1,000 test cases" and the FIDs would no longer be comparable. A committed JSON makes the
population an artifact you can point at, diff and cite.

**`load_cases` is stdlib-only on purpose.** A baseline adapter runs inside that baseline's own venv,
where `echosyn` does not import and torch may be a different major version. Reading the population
must cost nothing but `json`. Writing it needs MRFlow, so `dump_cases` imports lazily and is run
once, from the MRFlow venv.

The rows carry `archive`/`member` as absolute paths into the MR-RATE workspace, the same way
`evaluation/main.py`'s own manifests do. Those are the only fields that go stale if the raw data
moves; everything else is content.
"""

import argparse
import json


def load_cases(path):
    """The frozen population as a list of dicts. Stdlib only -- safe in any venv."""
    with open(path) as handle:
        return json.load(handle)["cases"]


def dump_cases(config_path, split, n_per_bucket, out):
    """Resolve the population through MRFlow's own selection and write it to `out`.

    Imports inside the function: `evaluation.main` pulls in torch, wandb and the generator stack
    (xformers included), which a caller that only wants to *read* a frozen list should never pay
    for.
    """
    from omegaconf import OmegaConf

    from echosyn.common.mrrate import list_series, sample_id
    from evaluation.main import conditioning_uid, select_cases

    config = OmegaConf.load(config_path)
    mri = config.mri
    series = list_series(mri.raw_root, split, 1, mri.get(f"max_series_{split}"), config.seed)
    if n_per_bucket:
        series = select_cases(series, n_per_bucket)

    cases = [{"case_id": sample_id(entry["study_uid"], entry["series_id"]),
              "bucket": f"{entry['modality']}__{entry['plane']}",
              "modality": entry["modality"], "plane": entry["plane"],
              "study_uid": entry["study_uid"], "series_id": entry["series_id"],
              "conditioning_uid": conditioning_uid(entry["study_uid"], entry["modality"],
                                                   entry["plane"]),
              "archive": entry["archive"], "member": entry["member"]}
             for entry in series]

    payload = {"split": split, "n_per_bucket": n_per_bucket, "seed": config.seed,
               "raw_root": mri.raw_root, "n_cases": len(cases), "cases": cases}
    with open(out, "w") as handle:
        json.dump(payload, handle, indent=1)

    buckets = {}
    for case in cases:
        buckets[case["bucket"]] = buckets.get(case["bucket"], 0) + 1
    print(f"{len(cases)} cases -> {out}")
    for bucket in sorted(buckets):
        print(f"  {bucket:24s} {buckets[bucket]}")
    return cases


def main():
    parser = argparse.ArgumentParser(description="Freeze the evaluation population to a file.")
    parser.add_argument("--config", required=True, help="An MRFlow config; only `mri` and `seed` "
                                                        "are read, so any of them will do.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--n_per_bucket", type=int, default=100,
                        help="100 is the 1,000-scored-case comparison population.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    dump_cases(args.config, args.split, args.n_per_bucket, args.out)


if __name__ == "__main__":
    main()
