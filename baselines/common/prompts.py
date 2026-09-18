"""The report text for every case of a frozen population, in one file.

    python -m baselines.common.prompts --cases baselines/cases-test-n100.json \
        --out $WS/baselines/prompts-test-n100.json

**Why this exists.** `cases-test-n100.json` says *which* series to generate, but a baseline also
needs the *report* -- and the reports live inside the MR-RATE tars, reachable only through
`echosyn.common.mrrate.read_report`. A baseline runs in its own venv, where `echosyn` does not
import, so without this it would have to reimplement the archive read. One exporter, run once from
the MRFlow venv, removes that duplication: after it, a baseline needs `json` and nothing else.

**This file is patient report text and must never be committed.** It defaults to the workspace for
that reason, and `--out` inside the repo is refused. `cases-test-n100.json` is committed because it
holds only ids and paths; this is the half that cannot be.

**Sections are kept separate, exactly as `report.json` stores them** -- `report`,
`clinical_information`, `technique`, `findings`, `impression`. Composing them is each baseline's
own job and differs per model: MRFlow joins `[FINDINGS]`/`[IMPRESSION]` behind an acquisition
prefix, R2V arm A does the same through its own formatter, R2V arm E encodes findings, impression
and acquisition as three separate tokens, Text2CT joins findings and impression into one string. Handing over a
pre-composed string would force one model's formatting onto the others, which is exactly what a
baseline must not inherit.

`impression` is empty for ~9% of studies. It is exported as an empty string rather than dropped, so
a consumer sees the absence instead of a missing key.
"""

import argparse
import json
import os


def load_prompts(path):
    """`{case_id: {section: text}}`. Stdlib only -- safe in any venv."""
    with open(path) as handle:
        return json.load(handle)["prompts"]


def dump_prompts(cases_path, out):
    """Read every case's report out of the MR-RATE archives and write them to `out`."""
    from echosyn.common.mrrate import read_report

    from baselines.common.cases import load_cases

    cases = load_cases(cases_path)
    reports, prompts = {}, {}
    for case in cases:
        study = case["study_uid"]
        if study not in reports:                 # one report per STUDY, many series per study
            reports[study] = read_report(case["archive"], study)
        report = reports[study]
        prompts[case["case_id"]] = {
            "study_uid": study,
            "modality": case["modality"],
            "plane": case["plane"],
            **{s: (report.get(s) or "").strip()
               for s in ("report", "clinical_information", "technique", "findings", "impression")},
        }

    with open(out, "w") as handle:
        json.dump({"cases": cases_path, "n_studies": len(reports), "n_cases": len(prompts),
                   "prompts": prompts}, handle, indent=1)

    empty = sum(1 for p in prompts.values() if not p["impression"])
    print(f"{len(prompts)} cases over {len(reports)} studies -> {out}")
    print(f"  empty impression: {empty} ({100*empty/len(prompts):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Export the reports for a frozen population.")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out", required=True, help="Outside the repo -- this is patient text.")
    args = parser.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.abspath(args.out).startswith(repo + os.sep):
        raise SystemExit(f"--out is inside the repo ({repo}). This file holds patient report text; "
                         f"write it to the workspace instead.")
    dump_prompts(args.cases, args.out)


if __name__ == "__main__":
    main()
