# 15篇KITTI论文引用注册表

- 检索/复核日期：2026-08-12
- 证据优先级：正式论文原文与补充材料 > KITTI官方页面 > 作者官方代码。博客、排行榜转述和论文解说不承载数值。
- 许可证只描述代码复用边界；论文思想仍须引用，未核清许可证时不复制代码。

| ID | 正式来源 | DOI/arXiv | 原文 | 作者官方代码与许可 | 本次核验重点 |
|---|---|---|---|---|---|
| K01 | Geiger, Lenz, Urtasun, *Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite*, CVPR 2012 | DOI `10.1109/CVPR.2012.6248074` | https://www.cvlibs.net/publications/Geiger2012CVPR.pdf | KITTI数据与benchmark为CC BY-NC-SA 3.0；https://www.cvlibs.net/datasets/kitti/ | 数据量、序列、hidden test、难度与现行评测变更；https://www.cvlibs.net/datasets/kitti/eval_object.php |
| K02 | Cai et al., *A Unified Multi-scale Deep Convolutional Neural Network for Fast Object Detection*, ECCV 2016 | DOI `10.1007/978-3-319-46493-0_22`；arXiv `1607.07155` | https://www.cvlibs.net/projects/autonomous_vision_survey/literature/Cai2016ECCV.pdf | https://github.com/zhaoweicai/mscnn ；UC Regents自定义许可仅允许教育、科研与非营利用途免费复制/修改/分发且须保留声明，商业使用需另行许可；第三方Caffe/Faster R-CNN分别保留其条款 | 多尺度proposal/detection、KITTI 2D消融与速度 |
| K03 | Xiang et al., *Subcategory-aware Convolutional Neural Networks for Object Proposals and Detection*, WACV 2017 | DOI `10.1109/WACV.2017.108`；arXiv `1604.04693` | https://yuxng.github.io/Papers/2017/xiang_wacv17.pdf | 未核验到对应作者官方实现/许可证 | 子类、图像金字塔、共享特征负结果、KITTI split/AP11 |
| K04 | Ren et al., *Accurate Single Stage Detector Using Recurrent Rolling Convolution*, CVPR 2017 | arXiv `1704.05776` | https://openaccess.thecvf.com/content_cvpr_2017/papers/Ren_Accurate_Single_Stage_CVPR_2017_paper.pdf | https://github.com/xiaohaoChen/rrc_detection ；BSD式上游条款，复用须保留归属 | 循环上下文、输出2–6合并的负消融、高IoU定位收益 |
| K05 | Tian et al., *Deep Learning Strong Parts for Pedestrian Detection*, ICCV 2015 | DOI `10.1109/ICCV.2015.221` | https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Tian_Deep_Learning_Strong_ICCV_2015_paper.pdf | 未核验到可归属的官方代码 | part models、遮挡、Caltech→KITTI迁移及不可比性 |
| K06 | Mao et al., *What Can Help Pedestrian Detection?*, CVPR 2017 | arXiv `1705.02757` | https://openaccess.thecvf.com/content_cvpr_2017/papers/Mao_What_Can_Help_CVPR_2017_paper.pdf | 未核验到论文对应官方代码/许可证 | HyperLearner辅助监督、分支压制、受控KITTI消融 |
| K07 | Han et al., *Wasserstein Loss based Deep Object Detection*, CVPR Workshops 2020 | DOI `10.1109/CVPRW50498.2020.00507`；arXiv未核验到 | https://openaccess.thecvf.com/content_CVPRW_2020/papers/w60/Han_Wasserstein_Loss-Based_Deep_Object_Detection_CVPRW_2020_paper.pdf | 未核验到作者官方实现 | Wasserstein分类代价、随机三折、非官方KITTI mAP边界 |
| K08 | Simon et al., *Complexer-YOLO: Real-Time 3D Object Detection and Tracking on Semantic Point Clouds*, CVPR Workshops 2019 | DOI `10.1109/CVPRW.2019.00158`；arXiv `1904.07537` | https://openaccess.thecvf.com/content_CVPRW_2019/papers/Autonomous%20Driving/Simon_Complexer_YOLO_Real-Time_3D_Object_Detection_and_Tracking_on_Semantic_CVPRW_2019_paper.pdf | 未核验到作者官方实现 | RGB语义+LiDAR 3D/BEV、hard filtering no-harm反例 |
| K09 | Brazil & Liu, *M3D-RPN: Monocular 3D Region Proposal Network for Object Detection*, ICCV 2019 | arXiv `1907.06038` | https://openaccess.thecvf.com/content_ICCV_2019/papers/Brazil_M3D-RPN_Monocular_3D_Region_Proposal_Network_for_Object_Detection_ICCV_2019_paper.pdf | https://github.com/brazgardian/M3D-RPN ；MIT | depth-aware convolution、双split、AP11与小类不稳定 |
| K10 | Ding et al., *Learning Depth-Guided Convolutions for Monocular 3D Object Detection*, CVPR 2020 | arXiv `1912.04799` | https://openaccess.thecvf.com/content_CVPR_2020/papers/Ding_Learning_Depth-Guided_Convolutions_for_Monocular_3D_Object_Detection_CVPR_2020_paper.pdf | https://github.com/dingmyu/D4LCN ；MIT，复用时仍须固定commit | 动态卷积、AP11/AP40并列、普通卷积+CL负消融 |
| K11 | Qian et al., *End-to-End Pseudo-LiDAR for Image-Based 3D Object Detection*, CVPR 2020 | arXiv `2004.03080` | https://openaccess.thecvf.com/content_CVPR_2020/papers/Qian_End-to-End_Pseudo-LiDAR_for_Image-Based_3D_Object_Detection_CVPR_2020_paper.pdf | https://github.com/mileyan/pseudo-LiDAR_e2e ，MIT | 可微坐标变换、梯度覆盖3/4/70%、量级重权、非加性训练 |
| K12 | Chen et al., *MonoRUn: Monocular 3D Object Detection by Reconstruction and Uncertainty Propagation*, CVPR 2021 | DOI `10.1109/CVPR46437.2021.01024`；arXiv `2103.12605` | https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_MonoRUn_Monocular_3D_Object_Detection_by_Reconstruction_and_Uncertainty_Propagation_CVPR_2021_paper.pdf | https://github.com/tjiiv-cprg/MonoRUn ，MIT | Robust-KL、E2E负结果、pose detach与校准所有权 |
| K13 | Lu et al., *Geometry Uncertainty Projection Network for Monocular 3D Object Detection*, ICCV 2021 | DOI `10.1109/ICCV48922.2021.00310`；arXiv `2107.13774` | https://openaccess.thecvf.com/content/ICCV2021/papers/Lu_Geometry_Uncertainty_Projection_Network_for_Monocular_3D_Object_Detection_ICCV_2021_paper.pdf | https://github.com/SuperMHP/GUPNet ，MIT | GeU/HTL、依赖调度、GeP×UnC负交互、宽/严IoU差异 |
| K14 | Ma et al., *Delving Into Localization Errors for Monocular 3D Object Detection*, CVPR 2021 | DOI `10.1109/CVPR46437.2021.00469`；arXiv `2103.16237` | https://openaccess.thecvf.com/content/CVPR2021/papers/Ma_Delving_Into_Localization_Errors_for_Monocular_3D_Object_Detection_CVPR_2021_paper.pdf | https://github.com/xinzhuma/monodle ，MIT | oracle误差定位、dimension-aware loss、far样本删除与数值冲突 |
| K15 | Zhang et al., *MonoDETR: Depth-guided Transformer for Monocular 3D Object Detection*, ICCV 2023 | arXiv `2203.13310` | https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_MonoDETR_Depth-guided_Transformer_for_Monocular_3D_Object_Detection_ICCV_2023_paper.pdf | https://github.com/ZrrSkywalker/MonoDETR ，MIT | depth-guided query、仅L2D matching、加3D loss降点/崩溃、约±1 AP波动 |

## 排除但保留的来源

- Song et al., *Small-scale Pedestrian Detection Based on Topological Line Localization and Temporal Feature Aggregation*, ECCV 2018：https://openaccess.thecvf.com/content_ECCV_2018/papers/Tao_Song_Small-scale_Pedestrian_Detection_ECCV_2018_paper.pdf 。主实验在Caltech/CityPersons；KITTI只有定性图与缺少完整协议的一处38.72，不能承担本清单的KITTI因果证据，故由K06 Mao/HyperLearner替换。

## 本地原始文件身份

- MonoDLE PDF：SHA256 `8C7C3F3005463E9539680454D2D26B3596343FB0AC01F02266B72D60A37BB0FF`。
- MonoDETR camera-ready PDF：SHA256 `1669F27AB9CB72DE6736E6691FF5786828C8D51E0C84947B05202493110788C4`。
- MonoDETR supplement：SHA256 `42D8C1C5EF21EDD15B42F52EAB4E66442230C5554EC55F7BB7CC84DBA6EEF03E`。
- 其余已下载PDF的逐文件身份保留在主技术卡对应A/L字段；没有身份的来源不得作为数值证据。
