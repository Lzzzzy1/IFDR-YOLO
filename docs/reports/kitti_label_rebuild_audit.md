# KITTI Label Rebuild Audit

Fixed 1242x375 normalization was compared with each image's actual dimensions.

- Images: 7481
- Training targets: 34856
- Affected targets: 6778
- Mean absolute normalized error: 0.0004954323
- Maximum absolute normalized error: 0.0144802619

| Image size | Images | Targets | Affected | Mean error | Max error |
|---|---:|---:|---:|---:|---:|
| 1224x370 | 770 | 2991 | 2991 | 0.0048036560 | 0.0144802619 |
| 1238x374 | 358 | 1880 | 1880 | 0.0009648124 | 0.0031936217 |
| 1241x376 | 296 | 1907 | 1907 | 0.0005701132 | 0.0021672695 |
| 1242x375 | 6057 | 28078 | 0 | 0.0000000000 | 0.0000000000 |
