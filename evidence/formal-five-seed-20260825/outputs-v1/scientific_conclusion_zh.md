# 修复后 Stage11 五种子正式结论

正式池固定为 seeds 0–4，比较 PLAIN_P2 与 DCLI，主指标为 KITTI 371 张 development 上 Pedestrian/Cyclist Moderate AP_R40 的非加权宏平均。

PLAIN_P2 五种子均值为 95.388640 AP，DCLI 为 94.370240 AP。按预注册符号 PLAIN_P2−DCLI，配对均值为 +1.018400 AP，95% 配对 t 区间为 [-0.405491, +2.442291]，PLAIN_P2 在 4/5 个 seed 上更高。

因此，修复后的正式五种子结果不支持“DCLI 稳定提升至少 1.1 AP”，也不支持五种子差异非零的结论。旧的 seed0 单点正增益只能保留为 preliminary/探索性观察，不能作为稳定提升写入论文。
