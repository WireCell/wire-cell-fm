"""Hydra's side of the framework: the modules supporting configuration handling for
Wire Cell FM (WCFM).

In particular:
 - schema.py: gives Hydra its defaults and its types. 
 - store.py: registers the schema with Hydra and the framework's five groups, 
            and the model group through the wcfm.config_schemas entry point.
 - io.py: determines how a resolved config is saved to disk for posterity. 
   It also owns the two derivations more than one package needs 
   (per_rank_batch_size, warmup_iters_from_epochs) and stays torch-free, so the
   config-only environment can check them and no two callers can disagree.

Note: these dataclasses are never instantiated. Hydra merges them into a DictConfig at
composition and they are gone; what gets constructed is the class each _target_ names, and
that happens elsewhere.

Check conf/README.md on how to build a configuration.
"""
