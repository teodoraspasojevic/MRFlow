"""Text2CT adapted to MR-RATE: fine-tuning, caching and the MR data contract.

Everything this baseline needs lives in this package. Nothing here imports `echosyn`, `evaluation`
or `lvfm`: the Text2CT stack runs in its own venv (a pre-v5 `transformers` for its vendored CLIP),
where MRFlow does not import. The MR-RATE preprocessing is therefore vendored rather than reused --
see `mrrate_data.py`, which names the MRFlow function each step was copied from.
"""
