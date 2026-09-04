# KITTI 15篇双人独立深读合并结论

- 日期：2026-08-12
- 主稿：`kitti_15_deep_yolo_agent_20260811.md`
- 独立复核稿：`kitti_15_partner_independent_20260811.md`
- 本文件只形成研究判断，不启动训练，不修改现有负/零结果。

## 1. 当前项目事实先于文献叙事

| 条件 | Pedestrian Moderate AP_R40 | Cyclist Moderate AP_R40 | 两类macro | 结论 |
|---|---:|---:|---:|---|
| clean P3–P5 YOLOv8m | 93.6667 | 99.8707 | 96.7687 | 同协议强基线 |
| clean plain-P2 | 93.2110 | 97.1429 | 95.1769 | 相对基线-1.5918，95% paired CI [-4.8269,+0.9269] |
| 30轮DCLI-only B（三seed均值） | — | — | B-C +0.3257 | CI [-1.5861,+2.2375]，不支持稳定正增益 |
| 30轮联合AB（三seed均值） | — | — | AB-C +0.1710 | CI [-1.5884,+1.9305]，不支持稳定正增益 |
| 联合相对DCLI | — | — | AB-B -0.1546 | 三seed全负；CI [-0.3184,+0.0091] |

371张已经参与路线选择，只能称development。文献中的任何正数都不能改变这些结果，更不能与其相加。

## 2. 15篇最稳定的共同规律

1. **受控单项收益通常是+0.5～+2.5 AP点，完整系统常见+1～+3点。** 更大的+4～+6往往同时改变多项、使用额外深度/传感器、换任务或换弱基线。
2. **高分辨率层同时带来细节和语义不足。** MS-CNN与RRC都需要hard-negative或深层上下文；这解释为什么普通P2不是天然增益。
3. **单项有效不能推出组合有效。** SubCNN共享层退化、RRC后期迭代退化、MonoRUn强loss上E2E为负、GUPNet GeP×UnC近零/负、MonoDETR把3D loss加入matching后下降或崩溃。
4. **必须区分coverage、localization、ranking。** RRC、GUPNet、MonoDLE在严格IoU上收益大而宽松IoU近零，说明它们主要修定位/排序而非新增召回。
5. **小类与远目标的单点非常脆弱。** M3D-RPN Cyclist从val约9–10掉到test约1；MonoDETR仓库自报约±1 AP3D训练波动；无seed/CI的小增益不可靠。
6. **总体AP可以通过牺牲难样本提高。** MonoDLE/MonoDETR过滤远距离训练目标；这不能写成far改善，也不能作为本项目主路线。
7. **公开先例边界很宽。** 普通P2/金字塔、多尺度融合、动态卷积、uncertainty loss、层级延迟、stop-gradient、稳定assignment、crop/zoom都不能再声称原创。

## 3. 我们自己的最大问题已经比文献假设更具体

plain-P2相对P3–P5的冻结预测显示：

- Pedestrian：TP `199→198`，FP `346→345`，FN `6→7`；mean `1-IoU`从0.1903改善到0.1665。
- Cyclist：TP `55→54`，FP `121→63`，FN `0→1`；mean `1-IoU`从0.1247改善到0.1043。
- small macro点估计+2.2797，但95% CI `[-9.0224,+12.8987]`；far macro -5.2455，CI同样很宽。

因此当前证据与“P2输出更干净、匹配框更准，但召回/score ranking/assignment链路丢失了极少数关键TP”相容。它尚不能证明TaskAlignedAssigner竞争就是根因，因为固定阈值TP/FP与matched IoU不能解释整条PR曲线，NMS、校准和单seed波动仍是替代解释。Cyclist Moderate有效GT约55个，丢1个就足以显著改变AP；这也解释了为什么small切片看似+2.28却不能加到总体。

## 4. 主路线：P2 tiny-safe assignment coverage rescue

### 假设

- **H1-A：** 在有至少一个合法P2候选的目标中，预注册的small/far Cyclist群体比near/large Cyclist对照具有更高的zero-P2-positive率；一个受限fallback可能恢复覆盖，同时保持全部原分配不变。
- **H0-A：** 两组zero-P2-positive率无实质差异，或相同缺陷遍布所有尺度；则P2专属assignment rescue没有支持，立即NO-GO。

### 先做的只读门

在3341 fit上按GT记录P2/P3/P4/P5候选数、正样本数、最高IoU、alignment rank与目标尺度；development只用于冻结结果评估，不用于找阈值。主群体固定为有合法P2候选的small/far Cyclist，对照固定为near/large Cyclist；估计按图像聚类的zero-P2-positive率差。只有点差至少10个绝对百分点、95% clustered CI下界高于0、且该缺陷不是所有尺度共同存在时才允许编码。

### 唯一最小改动

仅当H1-A通过时：对几何上属于P2负责范围、但原assigner给P2零正样本且存在合法候选的GT，提升**最高alignment的一个当前未分配P2候选**为正样本；保留每一个原始assignment，P3–P5分配、loss权重、neck、DCLI、fusion和NMS全部不变。每GT最多救一个，避免扩大正样本洪水或抢占其他GT候选。

### 30轮筛查门

- 同一3341/371、同一初始化与seed，同时跑30轮plain-P2 control与repair。
- 主比较：repair-control的Ped/Cyc Moderate macro AP_R40。
- 机制比较：zero-P2-positive率、TP/FN恢复、P2正样本数分布；并报Ped/Cyc、small/far、FP/FN、定位误差与延迟。
- GO需要同时满足冻结门：macro至少+1.0 AP点；10,000次image-cluster paired bootstrap的95% CI下界高于0；small/far的TP/FN与AP按预期改善且zero-P2-positive下降；Pedestrian overall、near与large各自伤害不超过0.5 AP；推理成本不变且训练开销完整报告。否则NO-GO；单seed只筛查，不写论文成功。

## 5. 备线假设：P2 candidate score ownership isolation（当前未授权）

这只是主路线NO-GO后的备选研究问题，当前没有预注册的效应量、CI门、跨层配对或融合契约，**不得据此编码或训练**。只有另行冻结fit-only raw pre-NMS ranking审计后，才能判断它是否与主路线互斥；zero-positive与ranking问题可能发生在不同GT上，不能靠文字假定互斥。

### 假设

- **H1-B：** P2候选定位良好，但浅层类别分数或DCLI/fusion校准改变其排序；让P3–P5语义分支保留类别/分配所有权，P2只输出detached localization-quality修正，可恢复TP而不重引入FP。
- **H0-B：** 漏失目标在raw pre-NMS输出中也没有候选，或隔离后TP/AP不恢复；则score ownership不是根因。

### 唯一最小改动与停止门

若未来审计通过，候选实现必须先解决标准Detect没有现成IoU-quality标量、P3–P5语义与P2定位之间需要跨层候选配对、以及固定NMS/融合是否可保持不变三个技术问题；不能先写一个quality标量假装问题已解决。MonoRUn的pose detach和MonoDETR的仅L2D matching是直接先例，因此隔离思想必须引用，不能称首创；若有贡献，只能是对特定P2道路小目标失配的实证与验证。

## 6. 为什么不把旧DCLI/fusion当主线

- DCLI-only三seed均值只有+0.3257且CI跨0。
- AB-C同样不显著；AB-B三seed全负，说明fusion没有给DCLI稳定增加价值。
- K12/K13/K15证明“不确定性/层级调度/隔离”都有直接先例；在未找出本项目独特错误位置前继续调schedule或叠adapter，容易变成结果导向搜索。
- 只有P2覆盖/排序审计明确把错误定位到共享校准路径后，才允许把ownership isolation作为备线；否则旧模块路线保持NO-GO。

## 7. 双稿交换后的差异与合并处理

| 主题 | 主稿独立发现 | 复核稿补充 | 合并结论 |
|---|---|---|---|
| K08 no-harm | 语义点云总体mAP上升 | hard filtering平均约+1.3却使Car Moderate BEV约-4.8 | 系统正收益必须逐类/难度检查伤害 |
| K10数字表达 | 逐表拆AP11/AP40与深度源 | 强调论文9.1%是相对比例，绝对仅+0.98 AP | 禁止把相对百分比写成AP点 |
| K11联合训练 | 梯度覆盖3/4/70%与重权 | soft quantization单独并不优于hard | 收益来自联合机制，不能拆成可相加模块 |
| K12校准 | Robust-KL上E2E/epistemic可负 | 复核确认小类与监督身份风险 | stop-gradient/校准隔离已有直接先例 |
| K13层级任务 | 宽IoU不升、严IoU大升；GeP×UnC近零/负 | 复核确认GradNorm失败及HTL时序 | 必须先判coverage/localization/ranking |
| K14远目标 | 删除>60m样本可提高aggregate | 复核确认阈值不单调 | far no-harm是强制门，不能靠删难样本 |
| K15匹配 | 加size/orientation降1.40/1.98，加depth崩溃 | 复核确认轻量结构优于更深/更宽 | 不稳定任务不得取得共享assignment所有权 |

两份独立稿没有需要二选一的核心事实冲突；差异主要是遗漏程度和强调点。上述合并结论保留更严格的协议/负结果解释。

## 8. 三个最强反方审查

1. **“96.77已经接近饱和，任何+1都是少数目标偶然。”** 回应：371只作development；报告GT/TP/FP/FN、image-paired CI和三seed；最终仍需未参与研发的confirmation或official test。
2. **“你只是重新实现公开assignment/stop-gradient。”** 回应：普通机制全部引用，不声称首创；只有在预注册错误位置、机制变化与AP共同验证后，才能主张特定P2交互的实证发现。
3. **“small正、总体负时挑small就是选择性报告。”** 回应：主指标始终是两类Moderate macro；small/far只是解释层，不能与总体相加，且必须报告near/large no-harm。
4. **“内部AP_R40未必等价官方devkit。”** 回应：在官方对账闭合前只称内部development结果，不用于官方SOTA比较。
5. **“30轮筛查不足。”** 回应：30轮只决定是否值得继续；正式结论仍需完整训练、多seed、paired CI、成本与独立确认。

## 9. 现实目标换算

- 当前plain-P2为95.1769；追平P3–P5需要相对plain-P2 `+1.5918`。
- 若最终相对P3–P5净`+1.0`，需要相对plain-P2 `+2.5918`。
- 净`+2.0`需要相对plain-P2 `+3.5918`。
- 净`+3.0`需要相对plain-P2 `+4.5918`，而当前开发集理论上距100仅3.2313点；极难且容易受小分母影响。

上述换算只是算术情景，不是新的验收口径。冻结门仍是：30轮repair-control达到至少+1 AP且完整统计门通过，才授权后续确认；最终plain-P2与完整系统性能门仍按既有预注册执行，不能用“相对P3–P5净+1”替代。若只读coverage门不通过，主路线不编码；若30轮未达到预注册门，保留负结果并停止，不通过换切片或堆模块追正数。
