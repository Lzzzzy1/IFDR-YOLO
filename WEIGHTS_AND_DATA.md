# Data and weight reconstruction boundary

## Dataset

This repository does not redistribute KITTI or BDD100K data. Obtain each
dataset from its official source under its own licence, keep the raw copy
read-only, and use the versioned split/configuration files in `configs/`.

The frozen formal protocol uses:

- fit subset: 3,341 images;
- internal development subset: 371 images;
- paired seeds: 0, 1, 2, 3, 4;
- input size: 640;
- batch size: 16;
- endpoint: epoch 30 `last.pt` only;
- evaluator: KITTI 2D AP_R40, with Moderate Pedestrian/Cyclist macro AP_R40 as
  the primary internal-development summary.

The 371-image development subset is not the official KITTI test set.

## Pretrained weight

The required upstream YOLOv8m pretrained weight is not committed. The frozen
SHA-256 recorded by the formal package is:

```text
5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5
```

Before training, obtain the appropriate upstream artifact, verify the hash,
and stop if it does not match the frozen identity. Trained PLAIN_P2 and DCLI
checkpoints are also not redistributed; the repository instead retains their
terminal receipts, source/config identities, raw metric outputs, and paper
statistics.

## Environment

The formal server evidence records Ultralytics `8.4.98`, PyTorch
`2.8.0+cu128`, and CUDA `12.8`. Earlier root-level infrastructure acceptance
was also exercised in a local PyTorch `2.5.1+cu121` environment. Reproducing
the formal numbers requires restoring the formal environment and exact frozen
assets, not merely any currently installed YOLO stack.
