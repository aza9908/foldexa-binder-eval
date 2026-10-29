# Local threshold overrides

Threshold profiles shipped with the package live in
`src/foldexa_binder_eval/profiles/` and are installed with the wheel.

This directory is searched **first**, so dropping a file named
`thresholds.<name>.yaml` here either adds a new profile or shadows a shipped
one of the same name, without reinstalling.

```yaml
# config/thresholds.house.yaml
extends: default
name: house
description: Calibrated on our own expressed designs.
thresholds:
  ipsae: 0.60
  delta_sasa: 1400
```

```bash
foldexa-binder-eval score design.cif --profile house
foldexa-binder-eval profiles          # confirms which file won
```

Keep project-specific thresholds here rather than editing the shipped
profiles, so `git diff` shows what your project changed and why.
