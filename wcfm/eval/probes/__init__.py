"""The probe suite: what a checkpoint's features are worth, measured on a fixed population.

Everything here is CPU-only and feature-file-only. No probe imports a dataset reader, a backbone
or warpconvnet: it reads an extraction off disk and scores it. That is what makes the expensive
GPU pass happen once per checkpoint while every metric stays cheap to re-run, even on a login
node, and it is why `wcfm eval submit` can fan the probes out to CPU slots.

`features.load_features` is the only door in. It presents the on-disk format -- mmapped blocks,
truth written once per eval set, pools drawn at extraction -- as the `Features` object the probe
bodies are written against, so a change of layout stops at that function.
"""
