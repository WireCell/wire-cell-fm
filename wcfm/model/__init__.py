"""The science: what the detector image is, and what the objective asks of it.

Everything under here knows the domain -- views, crops, masks, injection, teachers, heads,
terms -- and meets the framework at exactly one point, `wcfm.engine.protocol.TrainingModule`,
which `wcfm.model.modules.SslModule` implements. The framework packages may not import this
one, and `tests/test_import_graph.py` fails the build if one does. This one may import the
framework freely.

This `__init__` imports nothing heavy on purpose. `wcfm.model.config` is the entry point the
framework's config store loads, and it has to work in the config-only environment, which has
neither torch nor warpconvnet. The submodules that need them import them themselves.
"""
