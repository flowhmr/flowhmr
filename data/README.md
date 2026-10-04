# Data layout

```
data/
├── rendering_data/        # synthetic data rendering toolkit (Apache-2.0)
├── wild4k/                # Wild-4K evaluation clip list
├── smpl/                  # SMPL v1.1.0, full pipeline only (not bundled)
└── grpo_gt_pool/list.txt  # GRPO sample list (not bundled)
```

- [rendering_data/](rendering_data/README.md): renders SMPL-H motions in Blender and
  writes training samples. It is licensed under Apache-2.0, separately from the
  rest of the repository.
- [wild4k/](wild4k/README.md): clip IDs and categories of the 4182-clip Wild-4K set;
  videos are not distributed.
- `smpl/`: download and rename as in [install.md](../docs/install.md#41-body-models-manual).
- `grpo_gt_pool/list.txt`: one sample directory per line; see
  [resources.md](../docs/resources.md#sample-list-grpo).

Training shards (`tar_urls` in [configs/base/data.yml](../configs/base/data.yml))
can live anywhere. See [resources.md](../docs/resources.md) for the sample formats
and [training.md](../docs/training.md) for training.
