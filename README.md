# Awesome Industrial Anomaly Detection [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

We discuss public datasets and related studies in detail. Welcome to read our paper and make comments.

[Deep Industrial Image Anomaly Detection: A Survey (Machine Intelligence Research)](https://link.springer.com/article/10.1007/s11633-023-1459-z)

[IM-IAD: Industrial Image Anomaly Detection Benchmark in Manufacturing [TCYB 2024]](https://arxiv.org/abs/2301.13359)[[code]](https://github.com/M-3LAB/open-iad)[[中文]](https://blog.csdn.net/m0_63828250/article/details/136891730)

We will keep focusing on this field and updating relevant information.

Keywords: anomaly detection, anomaly segmentation, industrial image, defect detection

[[Main Page]](https://github.com/M-3LAB) [[Survey]](https://github.com/M-3LAB/awesome-industrial-anomaly-detection) [[Benchmark]](https://github.com/M-3LAB/open-iad) [[Result]](https://github.com/M-3LAB/IM-IAD)

🔥🔥🔥 Contributions to our repository are welcome. Feel free to categorize the papers.

---

🔥🔥🔥 Which MLLM performs best in industrial anomaly detection? Please refer to our recent research, which evaluates state-of-the-art models, including GPT-4o, Gemini-1.5, LLaVA-Next, and InternVL.

[2024.10.16] We are proud to announce the launch of MMAD, the first-ever comprehensive benchmark for Multimodal Large Language Models in Industrial Anomaly Detection! 🌟 [[Paper]](https://arxiv.org/abs/2410.09453) [[Code]](https://github.com/jam-cc/MMAD)  [[Data]](https://huggingface.co/datasets/jiang-cc/MMAD)

---
## Table of Contents
- [Awesome Industrial Anomaly Detection ](#awesome-industrial-anomaly-detection-)
  - [Table of Contents](#table-of-contents)
- [SOTA methods with code](#sota-methods-with-code)
- [Recommended Benchmarks](#recommended-benchmarks)
- [Recent research](#recent-research)
  - [AAAI 2025](#aaai-2025)
  - [NeurIPS 2024](#neurips-2024)
  - [ECCV 2024](#eccv-2024)
  - [ACM MM 2024](#acm-mm-2024)
  - [ICASSP 2024](#icassp-2024)
  - [CVPR 2024](#cvpr-2024)
  - [ICLR 2024](#iclr-2024)
  - [AAAI 2024](#aaai-2024)
  - [WACV 2024](#wacv-2024)
  - [NeurIPS 2023](#neurips-2023)
  - [LLM related](#llm-related)
  - [SAM segment anything](#sam-segment-anything)
  - [Others](#others)
  - [Medical (related)](#medical-related)
- [Paper Tree (Classification of representative methods)](#paper-tree-classification-of-representative-methods)
- [Timeline](#timeline)
- [Paper list for industrial image anomaly detection](#paper-list-for-industrial-image-anomaly-detection)
- [Related Survey, Benchmark, and Framework](#related-survey-benchmark-and-framework)
- [2 Unsupervised AD](#2-unsupervised-ad)
  - [2.1 Feature-Embedding-based Methods](#21-feature-embedding-based-methods)
    - [2.1.1 Teacher-Student](#211-teacher-student)
    - [2.1.2 One-Class Classification (OCC)](#212-one-class-classification-occ)
    - [2.1.3 Distribution-Map](#213-distribution-map)
    - [2.1.4 Memory Bank](#214-memory-bank)
    - [2.1.5 Vison Language AD](#215-vison-language-ad)
  - [2.2 Reconstruction-Based Methods](#22-reconstruction-based-methods)
    - [2.2.1 Autoencoder (AE)](#221-autoencoder-ae)
    - [2.2.2 Generative Adversarial Networks (GANs)](#222-generative-adversarial-networks-gans)
    - [2.2.3 Transformer](#223-transformer)
    - [2.2.4 Diffusion Model](#224-diffusion-model)
    - [2.2.5 Others](#225-others)
  - [2.3 Supervised AD](#23-supervised-ad)
    - [More Normal samples With (Less Abnormal Samples or Weak Labels)](#more-normal-samples-with-less-abnormal-samples-or-weak-labels)
    - [More Abnormal Samples](#more-abnormal-samples)
- [3 Other Research Direction](#3-other-research-direction)
  - [3.1 Zero/Few-Shot AD](#31-zerofew-shot-ad)
    - [Zero-Shot AD](#zero-shot-ad)
    - [Few-Shot AD](#few-shot-ad)
  - [3.2 Noisy AD](#32-noisy-ad)
  - [3.3 Anomaly Synthetic](#33-anomaly-synthetic)
  - [3.4 RGBD AD](#34-rgbd-ad)
  - [3.5 3D AD](#35-3d-ad)
  - [3.6 Continual AD](#36-continual-ad)
  - [3.7 Uniform/Multi-Class AD](#37-uniformmulti-class-ad)
  - [3.8 Logical AD](#38-logical-ad)
  - [Other settings](#other-settings)
    - [TTT binary segmentation](#ttt-binary-segmentation)
    - [MoE with TTA](#moe-with-tta)
    - [Adversary Attack](#adversary-attack)
    - [Defect Classification](#defect-classification)
- [4 Dataset](#4-dataset)
  - [BibTex Citation](#bibtex-citation)
  - [Star History](#star-history)


# SOTA methods with code

|  Title  |   Venue  |   Date   |   Code   |   topic   |
|:--------|:--------:|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/hq-deng/RD4AD.svg?style=social&label=Star) <br> [**Anomaly Detection via Reverse Distillation from One-Class Embedding**](https://openaccess.thecvf.com/content/CVPR2022/html/Deng_Anomaly_Detection_via_Reverse_Distillation_From_One-Class_Embedding_CVPR_2022_paper.html) <br> | CVPR | 2022 | [Github](https://github.com/hq-deng/RD4AD) | Teacher-Student |
| ![Star](https://img.shields.io/github/stars/tientrandinh/Revisiting-Reverse-Distillation.svg?style=social&label=Star) <br> [**Revisiting Reverse Distillation for Anomaly Detection**](https://openaccess.thecvf.com/content/CVPR2023/html/Tien_Revisiting_Reverse_Distillation_for_Anomaly_Detection_CVPR_2023_paper.html) <br> | CVPR | 2023 | [Github](https://github.com/tientrandinh/Revisiting-Reverse-Distillation) | Teacher-Student |
| ![Star](https://img.shields.io/github/stars/DonaldRR/SimpleNet.svg?style=social&label=Star) <br> [**SimpleNet: A Simple Network for Image Anomaly Detection and Localization**](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.html) <br> | CVPR | 2023 | [Github](https://github.com/DonaldRR/SimpleNet) | One-Class-Classification |
| ![Star](https://img.shields.io/github/stars/gudovskiy/cflow-ad.svg?style=social&label=Star) <br> [**Real-time unsupervised anomaly detection with localization via conditional normalizing flows**](https://openaccess.thecvf.com/content/WACV2022/html/Gudovskiy_CFLOW-AD_Real-Time_Unsupervised_Anomaly_Detection_With_Localization_via_Conditional_Normalizing_WACV_2022_paper.html) <br> | WACV | 2022 | [Github](https://github.com/gudovskiy/cflow-ad) | Distribution Map |
| ![Star](https://img.shields.io/github/stars/gasharper/PyramidFlow.svg?style=social&label=Star) <br> [**PyramidFlow: High-Resolution Defect Contrastive Localization using Pyramid Normalizing Flow**](https://openaccess.thecvf.com/content/CVPR2023/html/Lei_PyramidFlow_High-Resolution_Defect_Contrastive_Localization_Using_Pyramid_Normalizing_Flow_CVPR_2023_paper.html) <br> | CVPR | 2023 | [Github](https://github.com/gasharper/PyramidFlow) | Distribution Map |
| ![Star](https://img.shields.io/github/stars/amazon-science/patchcore-inspection.svg?style=social&label=Star) <br> [**Towards total recall in industrial anomaly detection**](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html) <br> | CVPR | 2022 | [Github](https://github.com/amazon-science/patchcore-inspection) | Memory-bank |
| ![Star](https://img.shields.io/github/stars/wogur110/PNI_Anomaly_Detection.svg?style=social&label=Star) <br> [**PNI: Industrial Anomaly Detection using Position and Neighborhood Information**](https://openaccess.thecvf.com/content/ICCV2023/html/Bae_PNI__Industrial_Anomaly_Detection_using_Position_and_Neighborhood_Information_ICCV_2023_paper.html) <br> | ICCV | 2023 | [Github](https://github.com/wogur110/PNI_Anomaly_Detection) | Memory-bank |
| ![Star](https://img.shields.io/github/stars/vitjanz/draem.svg?style=social&label=Star) <br> [**Draem-a discriminatively trained reconstruction embedding for surface anomaly detection**](https://openaccess.thecvf.com/content/ICCV2021/html/Zavrtanik_DRAEM_-_A_Discriminatively_Trained_Reconstruction_Embedding_for_Surface_Anomaly_ICCV_2021_paper.html) <br> | ICCV | 2021 | [Github](https://github.com/vitjanz/draem) | Reconstruction-based |
| ![Star](https://img.shields.io/github/stars/VitjanZ/DSR_anomaly_detection.svg?style=social&label=Star) <br> [**DSR: A dual subspace re-projection network for surface anomaly detection**](https://link.springer.com/chapter/10.1007/978-3-031-19821-2_31) <br> | ECCV | 2022 | [Github](https://github.com/VitjanZ/DSR_anomaly_detection) | Reconstruction-based |
| ![Star](https://img.shields.io/github/stars/zhangzjn/ocr-gan.svg?style=social&label=Star) <br> [**Omni-frequency Channel-selection Representations for Unsupervised Anomaly Detection**](https://ieeexplore.ieee.org/abstract/document/10192551/) <br> | TIP | 2023 | [Github](https://github.com/zhangzjn/ocr-gan) | Reconstruction-based |
| ![Star](https://img.shields.io/github/stars/cnulab/RealNet.svg?style=social&label=Star) <br> [**RealNet: A Feature Selection Network with Realistic Synthetic Anomaly for Anomaly Detection**](https://arxiv.org/abs/2403.05897) <br> | CVPR | 2024 | [Github](https://github.com/cnulab/RealNet) | Reconstruction-based |
| ![Star](https://img.shields.io/github/stars/MediaBrain-SJTU/RegAD.svg?style=social&label=Star) <br> [**Registration based few-shot anomaly detection**](https://link.springer.com/chapter/10.1007/978-3-031-20053-3_18) <br> | ECCV | 2022 | [Github](https://github.com/MediaBrain-SJTU/RegAD) | Few Shot |
| ![Star](https://img.shields.io/github/stars/CASIA-IVA-Lab/AnomalyGPT.svg?style=social&label=Star) <br> [**AnomalyGPT: Detecting Industrial Anomalies using Large Vision-Language Models**](https://arxiv.org/abs/2308.15366) <br> | AAAI | 2024 | [Github](https://github.com/CASIA-IVA-Lab/AnomalyGPT) | Few Shot |
| ![Star](https://img.shields.io/github/stars/Choubo/DRA.svg?style=social&label=Star) <br> [**Catching Both Gray and Black Swans: Open-set Supervised Anomaly Detection**](https://openaccess.thecvf.com/content/CVPR2022/html/Ding_Catching_Both_Gray_and_Black_Swans_Open-Set_Supervised_Anomaly_Detection_CVPR_2022_paper.html) <br> | CVPR | 2022 | [Github](https://github.com/Choubo/DRA) | Few abnormal samples |
| ![Star](https://img.shields.io/github/stars/xcyao00/BGAD.svg?style=social&label=Star) <br> [**Explicit Boundary Guided Semi-Push-Pull Contrastive Learning for Supervised Anomaly Detection**](https://openaccess.thecvf.com/content/CVPR2023/html/Yao_Explicit_Boundary_Guided_Semi-Push-Pull_Contrastive_Learning_for_Supervised_Anomaly_Detection_CVPR_2023_paper.html) <br> | CVPR | 2023 | [Github](https://github.com/xcyao00/BGAD) | Few abnormal samples |
| ![Star](https://img.shields.io/github/stars/tianyu0207/IGD.svg?style=social&label=Star) <br> [**Deep one-class classification via interpolated gaussian descriptor**](https://ojs.aaai.org/index.php/AAAI/article/view/19915) <br> | AAAI | 2022 | [Github](https://github.com/tianyu0207/IGD) | Noisy AD |
| ![Star](https://img.shields.io/github/stars/TencentYoutuResearch/AnomalyDetection-SoftPatch.svg?style=social&label=Star) <br> [**SoftPatch: Unsupervised Anomaly Detection with Noisy Data**](https://proceedings.neurips.cc/paper_files/paper/2022/hash/637a456d89289769ac1ab29617ef7213-Abstract-Conference.html) <br> | NeurIPS | 2022 | [Github](https://github.com/TencentYoutuResearch/AnomalyDetection-SoftPatch) | Noisy AD |
| ![Star](https://img.shields.io/github/stars/DeclanMcIntosh/InReaCh.svg?style=social&label=Star) <br> [**Inter-Realization Channels: Unsupervised Anomaly Detection Beyond One-Class Classification**](https://openaccess.thecvf.com/content/ICCV2023/html/McIntosh_Inter-Realization_Channels_Unsupervised_Anomaly_Detection_Beyond_One-Class_Classification_ICCV_2023_paper.html) <br> | ICCV | 2023 | [Github](https://github.com/DeclanMcIntosh/InReaCh) | Noisy AD |
| ![Star](https://img.shields.io/github/stars/shirowalker/UCAD.svg?style=social&label=Star) <br> [**Unsupervised Continual Anomaly Detection with Contrastively-learned Prompt**](https://ojs.aaai.org/index.php/AAAI/article/view/28153) <br> | AAAI | 2024 | [Github](https://github.com/shirowalker/UCAD) | Continual AD |
| ![Star](https://img.shields.io/github/stars/zhiyuanyou/UniAD.svg?style=social&label=Star) <br> [**A Unified Model for Multi-class Anomaly Detection**](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1d774c112926348c3e25ea47d87c835b-Abstract-Conference.html) <br> | NeurIPS | 2022 | [Github](https://github.com/zhiyuanyou/UniAD) | Multi-class unified |
| ![Star](https://img.shields.io/github/stars/RuiyingLu/HVQ-Trans.svg?style=social&label=Star) <br> [**Hierarchical Vector Quantized Transformer for Multi-class Unsupervised Anomaly Detection**](https://openreview.net/pdf?id=clJTNssgn6) <br> | NeurIPS | 2023 | [Github](https://github.com/RuiyingLu/HVQ-Trans) | Multi-class unified |
| ![Star](https://img.shields.io/github/stars/nomewang/M3DM.svg?style=social&label=Star) <br> [**Multimodal Industrial Anomaly Detection via Hybrid Fusion**](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Multimodal_Industrial_Anomaly_Detection_via_Hybrid_Fusion_CVPR_2023_paper.html) <br> | CVPR | 2023 | [Github](https://github.com/nomewang/M3DM) | RGBD |
| ![Star](https://img.shields.io/github/stars/M-3LAB/Real3D-AD.svg?style=social&label=Star) <br> [**Real3D-AD: A Dataset of Point Cloud Anomaly Detection**](https://openreview.net/pdf?id=zGthDp4yYe) <br> | NeurIPS | 2023 | [Github](https://github.com/M-3LAB/Real3D-AD) | Point Cloud |
| ![Star](https://img.shields.io/github/stars/hq-deng/AnoVL.svg?style=social&label=Star) <br> [**AnoVL: Adapting Vision-Language Models for Unified Zero-shot Anomaly Localization**](https://arxiv.org/abs/2308.15939) <br> | arxiv | 2023 | [Github](https://github.com/hq-deng/AnoVL) | Zero Shot |
| ![Star](https://img.shields.io/github/stars/caoyunkang/GroundedSAM-zero-shot-anomaly-detection.svg?style=social&label=Star) <br> [**Segment Any Anomaly without Training via Hybrid Prompt Regularization**](https://arxiv.org/abs/2305.10724) <br> | arxiv | 2023 | [Github](https://github.com/caoyunkang/GroundedSAM-zero-shot-anomaly-detection) | Zero Shot |
| ![Star](https://img.shields.io/github/stars/oopil/PSAD_logical_anomaly_detection.svg?style=social&label=Star) <br> [**PSAD: Few Shot Part Segmentation Reveals Compositional Logic for Industrial Anomaly Detection**](https://ojs.aaai.org/index.php/AAAI/article/view/28703) <br> | AAAI | 2024 | [Github](https://github.com/oopil/PSAD_logical_anomaly_detection) | Logical/Few Shot |
| ![Star](https://img.shields.io/github/stars/YoojLee/Uniformaly.svg?style=social&label=Star) <br> [**UniFormaly: Towards Task-Agnostic Unified Framework for Visual Anomaly Detection**](https://arxiv.org/abs/2307.12540) <br> | arxiv | 2023 | [Github](https://github.com/YoojLee/Uniformaly) | Multi-class unified |

# Recommended Benchmarks
|  Title  |   Venue  |   Date   |   Code   |   topic   |
|:--------|:--------:|:--------:|:--------:|:--------:|
| ![Star](https://img.shields.io/github/stars/openvinotoolkit/anomalib.svg?style=social&label=Star) <br> [**Anomalib: A Deep Learning Library for Anomaly Detection**](https://ieeexplore.ieee.org/abstract/document/9897283/) <br> | ICIP | 2022 | [Github](https://github.com/openvinotoolkit/anomalib) | Benchmark |
| ![Star](https://img.shields.io/github/stars/M-3LAB/open-iad.svg?style=social&label=Star) <br> [**IM-IAD: Industrial Image Anomaly Detection Benchmark in Manufacturing**](https://arxiv.org/abs/2301.13359) <br> | TCYB | 2024 | [Github](https://github.com/M-3LAB/open-iad) | Benchmark |
| ![Star](https://img.shields.io/github/stars/zhangzjn/ader.svg?style=social&label=Star) <br> [**ADer: A Comprehensive Benchmark for Multi-class Visual Anomaly Detection**](http://arxiv.org/pdf/2406.03262v1) <br> | arxiv | 2024 | [Github](https://github.com/zhangzjn/ader) | Benchmark |
| ![Star](https://img.shields.io/github/stars/jam-cc/MMAD.svg?style=social&label=Star) <br> [**MMAD: The First-Ever Comprehensive Benchmark for Multimodal Large Language Models in Industrial Anomaly Detection**](https://arxiv.org/abs/2410.09453) <br> | arxiv | 2024 | [Github](https://github.com/jam-cc/MMAD) | Benchmark |



+ Anomaly Detection on MVTec AD [[paper with code]](https://paperswithcode.com/sota/anomaly-detection-on-mvtec-ad)
+ Anomaly Detection on VisA [[paper with code]](https://paperswithcode.com/sota/anomaly-detection-on-visa)
+ Anomaly Detection on MVTec LOCO AD [[paper with code]](https://paperswithcode.com/sota/anomaly-detection-on-mvtec-loco-ad)
+ Anomaly Detection on MVTec 3D-AD [[paper with code]](https://paperswithcode.com/sota/rgb-3d-anomaly-detection-and-segmentation-on)
+ Anomaly Detection Datasets and Benchmarks [[paper with code]](https://paperswithcode.com/task/anomaly-detection)

# Recent research

## AAAI 2025
+ MVREC: A General Few-shot Defect Classification Model Using Multi-View Region-Context [[AAAI 2025]](https://arxiv.org/abs/2412.16897)
+ Revisiting Multimodal Fusion for 3D Anomaly Detection from an Architectural Perspective [[AAAI 2025]](https://arxiv.org/abs/2412.17297)
+ KAG-prompt: Kernel-Aware Graph Prompt Learning for Few-Shot Anomaly Detection [[AAAI 2025]](https://arxiv.org/abs/2412.17619)[[code]](https://github.com/CVL-hub/KAG-prompt)
+ FiCo: Filter or Compensate: Towards Invariant Representation from Distribution Shift for Anomaly Detection [[AAAI 2025]](https://arxiv.org/abs/2412.10115)[[code]](https://github.com/znchen666/FiCo)
+ CKAAD: Boosting Fine-Grained Visual Anomaly Detection with Coarse-Knowledge-Aware Adversarial Learning [[AAAI 2025]](https://arxiv.org/abs/2412.12850)[[code]](https://github.com/Faustinaqq/CKAAD)
+ CNC: Cross-modal Normality Constraint for Unsupervised Multi-class Anomaly Detection [[AAAI 2025]](https://arxiv.org/abs/2501.00346)[[code]](https://github.com/cvddl/CNC)
+ Look Inside for More: Internal Spatial Modality Perception for 3D Anomaly Detection [[AAAI 2025]](https://arxiv.org/abs/2412.13461)
+ Unlocking the Potential of Reverse Distillation for Anomaly Detection [[AAAI 2025]](https://arxiv.org/abs/2412.07579)[[code]](https://github.com/hito2448/URD)
+ Promptable Anomaly Segmentation with SAM Through Self-Perception Tuning [[AAAI 2025]](https://arxiv.org/abs/2411.17217) [[code]](https://github.com/THU-MIG/SAM-SPT)

## NeurIPS 2024
+ MambaAD: Exploring State Space Models for Multi-class Unsupervised Anomaly Detection [[NeurIPS 2024]](https://arxiv.org/abs/2404.06564)[[code]](https://lewandofskee.github.io/projects/MambaAD/)
+ PointAD: Comprehending 3D Anomalies from Points and Pixels for Zero-shot 3D Anomaly Detection [[NeurIPS 2024]](https://arxiv.org/abs/2410.00320)[[code]](https://github.com/zqhang/PointAD)
+ CableInspect-AD: An Expert-Annotated Anomaly Detection Dataset [[NeurIPS 2024]](https://arxiv.org/abs/2409.20353)[[data]](https://mila-iqia.github.io/cableinspect-ad/)
+ ResAD: A Simple Framework for Class Generalizable Anomaly Detection [[NeurIPS 2024]](https://arxiv.org/abs/2410.20047)[[code]](https://github.com/xcyao00/ResAD)
<!-- + MetaUAS: Universal Anomaly Segmentation with One-Prompt Meta-Learning [[NeurIPS 2024]](https://openreview.net/forum?id=4jegYnUMHb&referrer=%5Bthe%20profile%20of%20Bin-Bin%20Gao%5D(%2Fprofile%3Fid%3D~Bin-Bin_Gao1))[[code]](https://github.com/gaobb/MetaUAS)-->

## ECCV 2024
+ R3D-AD: Reconstruction via Diffusion for 3D Anomaly Detection [[ECCV 2024]](https://arxiv.org/abs/2407.10862)[[homepage]](https://zhouzheyuan.github.io/r3d-ad)
+ An Incremental Unified Framework for Small Defect Inspection [[ECCV2024]](https://arxiv.org/abs/2312.08917v2)[[code]](https://github.com/jqtangust/IUF)
+ Learning Unified Reference Representation for Unsupervised Multi-class Anomaly Detection [[ECCV 2024]](https://arxiv.org/abs/2403.11561)[[code]](https://github.com/hlr7999/RLR)
+ Self-supervised Feature Adaptation for 3D Industrial Anomaly Detection [[ECCV 2024]](https://arxiv.org/abs/2401.03145)
+ Learning to Detect Multi-class Anomalies with Just One Normal Image Prompt [[ECCV 2024]](https://csgaobb.github.io/Pub_files/ECCV2024_OneNIP_CR_Full_0725_Mobile.pdf)[[code]](https://github.com/gaobb/OneNIP)
+ Few-Shot Anomaly-Driven Generation for Anomaly Classification and Segmentation [[ECCV 2024]](https://csgaobb.github.io/Pub_files/ECCV2024_AnoGen_CR_0730_Mobile.pdf)[[code]](https://github.com/gaobb/AnoGen)
+ AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for Zero-Shot Anomaly Detection [[ECCV 2024]](https://arxiv.org/abs/2407.15795)[[code]](https://github.com/caoyunkang/AdaCLIP)
+ GLAD: Towards Better Reconstruction with Global and Local Adaptive Diffusion Models for Unsupervised Anomaly Detection [[ECCV 2024]](https://arxiv.org/abs/2406.07487)[[code]](https://github.com/hyao1/GLAD)
+ GeneralAD: Anomaly Detection Across Domains by Attending to Distorted Features [[ECCV 2024]](https://arxiv.org/abs/2407.12427)[[code]](https://github.com/LucStrater/GeneralAD)
+ VCP-CLIP: A visual context prompting model for zero-shot anomaly segmentation [[ECCV 2024]](https://arxiv.org/abs/2407.12276)[[code]](https://github.com/xiaozhen228/VCP-CLIP)
+ A Unified Anomaly Synthesis Strategy with Gradient Ascent for Industrial Anomaly Detection and Localization [[ECCV 2024]](https://arxiv.org/abs/2407.09359)[[code]](https://github.com/cqylunlun/GLASS)
+ Hierarchical Gaussian Mixture Normalizing Flow Modeling for Unified Anomaly Detection [[ECCV 2024]](https://arxiv.org/abs/2403.13349)[[code]](https://github.com/xcyao00/HGAD)
+ TransFusion -- A Transparency-Based Diffusion Model for Anomaly Detection [[ECCV 2024]](https://arxiv.org/abs/2311.09999)[[code]](https://github.com/MaticFuc/ECCV_TransFusion)
+ Continuous Memory Representation for Anomaly Detection [[ECCV 2024]](https://arxiv.org/abs/2402.18293)[[homepage]](https://tae-mo.github.io/crad/)[[code]](https://github.com/tae-mo/CRAD)
+ Defect Spectrum: A Granular Look of Large-Scale Defect Datasets with Rich Semantics [[ECCV 2024]](https://openreview.net/forum?id=RLhS1TrjK3)[[data]](https://github.com/EnVision-Research/Defect_Spectrum)
+ AD3: Introducing a score for Anomaly Detection Dataset Difficulty assessment using VIADUCT dataset [[ECCV 2024]](https://eccv.ecva.net/virtual/2024/poster/2287)
+ Learning Diffusion Models for Multi-View Anomaly Detection [[ECCV 2024]](https://eccv2024.ecva.net/virtual/2024/poster/1911)
+ MoEAD: A Parameter-efficient Model for Multi-class Anomaly Detection [[ECCV 2024]](https://eccv2024.ecva.net/virtual/2024/poster/2653)[[code]](https://github.com/TheStarOfMSY/MoEAD)
+ Unsupervised, Online and On-The-Fly Anomaly Detection For Non-Stationary Image Distributions [[ECCV 2024]](https://eccv2024.ecva.net/virtual/2024/poster/2289)[[code]](https://github.com/DeclanMcIntosh/Online_InReaCh)
+ Tackling Structural Hallucination in Image Translation with Local Diffusion [[ECCV 2024 oral]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10498.pdf)[[code]](https://github.com/edshkim98/LocalDiffusion-Hallucination)

## ACM MM 2024
+ FiLo: Zero-Shot Anomaly Detection by Fine-Grained Description and High-Quality Localization [[ACM MM 2024]](https://arxiv.org/abs/2404.13671)[[code]](https://github.com/CASIA-IVA-Lab/FiLo)
+ Dual-Modeling Decouple Distillation for Unsupervised Anomaly Detection [[ACM MM 2024]](https://arxiv.org/abs/2408.03888)
+ FOCT: Few-shot Industrial Anomaly Detection with Foreground-aware Online Conditional Transport [[ACM MM 2024]](https://dl.acm.org/doi/10.1145/3664647.3680771)
+ Towards High-resolution 3D Anomaly Detection via Group-Level Feature Contrastive Learning [[ACM MM 2024]](https://arxiv.org/abs/2408.04604)[[code]](https://github.com/M-3LAB/Group3AD)

## ICASSP 2024
+ Implicit Foreground-Guided Network for Anomaly Detection and Localization [[ICASSP 2024]](https://ieeexplore.ieee.org/abstract/document/10446952)
+ Neural Network Training Strategy To Enhance Anomaly Detection Performance: A Perspective On Reconstruction Loss Amplification [[ICASSP 2024]](https://ieeexplore.ieee.org/document/10446942)
+ Patch-Wise Augmentation for Anomaly Detection and Localization [[ICASSP 2024]](https://ieeexplore.ieee.org/document/10446994)
+ A Reconstruction-Based Feature Adaptation for Anomaly Detection with Self-Supervised Multi-Scale Aggregation [[ICASSP 2024]](https://ieeexplore.ieee.org/document/10446766)
+ Feature-Constrained and Attention-Conditioned Distillation Learning for Visual Anomaly Detection [[ICASSP 2024]](https://ieeexplore.ieee.org/document/10448432)
+ CAGEN: Controllable Anomaly Generator using Diffusion Model [[ICASSP 2024]](https://ieeexplore.ieee.org/document/10447663)
+ Mixed-Attention Auto Encoder for Multi-Class Industrial Anomaly Detection [[ICASSP 2024]](https://ieeexplore.ieee.org/document/10446794)

## CVPR 2024
+ Text-Guided Variational Image Generation for Industrial Anomaly Detection and Segmentation [[CVPR 2024]](https://arxiv.org/abs/2403.06247)[[code]](https://github.com/MingyuLee82/TGI_AD_v1)
+ RealNet: A Feature Selection Network with Realistic Synthetic Anomaly for Anomaly Detection [[CVPR 2024]](https://arxiv.org/abs/2403.05897)[[code]](https://github.com/cnulab/RealNet)
+ Toward Generalist Anomaly Detection via In-context Residual Learning with Few-shot Sample Prompts [[CVPR 2024]](https://arxiv.org/abs/2403.06495)[[code]](https://github.com/mala-lab/InCTRL)
+ Multimodal Industrial Anomaly Detection by Crossmodal Feature Mapping [[CVPR 2024]](https://arxiv.org/abs/2312.04521)
+ Towards Scalable 3D Anomaly Detection and Localization: A Benchmark via 3D Anomaly Synthesis and A Self-Supervised Learning Network [[CVPR 2024]](https://arxiv.org/abs/2311.14897)[[code]](https://github.com/Chopper-233/Anomaly-ShapeNet)
+ Real-IAD: A Real-World Multi-view Dataset for Benchmarking Versatile Industrial Anomaly Detection [[CVPR 2024]](https://arxiv.org/abs/2403.12580)[[code]](https://github.com/TencentYoutuResearch/AnomalyDetection_Real-IAD)[[data]](https://realiad4ad.github.io/Real-IAD/)
+ Long-Tailed Anomaly Detection with Learnable Class Names [[CVPR 2024]](https://arxiv.org/abs/2403.20236)[[data split]](https://zenodo.org/records/10854201)
+ PromptAD: Learning Prompts with only Normal Samples for Few-Shot Anomaly Detection [[CVPR 2024]](https://arxiv.org/abs/2404.05231)[[code]](https://github.com/FuNz-0/PromptAD)
+ Supervised Anomaly Detection for Complex Industrial Images [[CVPR 2024]](https://openaccess.thecvf.com/content/CVPR2024/html/Baitieva_Supervised_Anomaly_Detection_for_Complex_Industrial_Images_CVPR_2024_paper.html)[[code]](https://github.com/abc-125/segad)
+ Anomaly Heterogeneity Learning for Open-set Supervised Anomaly Detection [[CVPR 2024]](https://arxiv.org/abs/2310.12790)[[code]](https://github.com/mala-lab/AHL)
+ Prompt-enhanced Multiple Instance Learning for Weakly Supervised Anomaly Detection [[CVPR 2024]](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Prompt-Enhanced_Multiple_Instance_Learning_for_Weakly_Supervised_Video_Anomaly_Detection_CVPR_2024_paper.html)[[code]](https://github.com/Junxi-Chen/PE-MIL)
+ Looking 3D: Anomaly Detection with 2D-3D Alignment [[CVPR 2024]](https://openaccess.thecvf.com/content/CVPR2024/html/Bhunia_Looking_3D_Anomaly_Detection_with_2D-3D_Alignment_CVPR_2024_paper.html)[[homepage]](https://groups.inf.ed.ac.uk/vico/research/Looking3D)[[code]](https://github.com/VICO-UoE/Looking3D)
+ CVPRW: VAND 2.0: Visual Anomaly and Novelty Detection - 2nd Edition [[Challenge and Call for Papers]](https://sites.google.com/view/vand-2-0-cvpr-2024/home)
+ Divide and Conquer: High-Resolution Industrial Anomaly Detection via Memory Efficient Tiled Ensemble [[CVPR 24 Visual Anomaly Detection Workshop]](https://arxiv.org/abs/2403.04932)[[homepage]](https://summerofcode.withgoogle.com/archive/2023/projects/WUSjdxGl)

## ICLR 2024
+ AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection [[ICLR 2024]](https://openreview.net/forum?id=buC4E91xZE)[[code]](https://github.com/zqhang/AnomalyCLIP)
+ MuSc: Zero-Shot Industrial Anomaly Classification and Segmentation with Mutual Scoring of the Unlabeled Images[[ICLR 2024]](https://openreview.net/forum?id=AHgc5SMdtd)[[code]](https://github.com/xrli-U/MuSc)

## AAAI 2024
+ Rethinking Reverse Distillation for Multi-Modal Anomaly Detection [[AAAI 2024]](https://ojs.aaai.org/index.php/AAAI/article/view/28687)
+ Unsupervised Continual Anomaly Detection with Contrastively-learned Prompt [[AAAI 2024]](https://ojs.aaai.org/index.php/AAAI/article/view/28153)[[code]](https://github.com/shirowalker/UCAD)
+ Few Shot Part Segmentation Reveals Compositional Logic for Industrial Anomaly Detection [[AAAI 2024]](https://ojs.aaai.org/index.php/AAAI/article/view/28703)[[code]](https://github.com/oopil/PSAD_logical_anomaly_detection)
+ DiAD: A Diffusion-based Framework for Multi-class Anomaly Detection [[AAAI 2024]](https://ojs.aaai.org/index.php/AAAI/article/view/28690)[[code]](https://lewandofskee.github.io/projects/diad)
+ Generating and Reweighting Dense Contrastive Patterns for Unsupervised Anomaly Detection [[AAAI 2024]](https://ojs.aaai.org/index.php/AAAI/article/view/27910)
+ AnomalyDiffusion: Few-Shot Anomaly Image Generation with Diffusion Model [[AAAI 2024]](https://ojs.aaai.org/index.php/AAAI/article/view/28696)[[code]](https://github.com/sjtuplayer/anomalydiffusion)
+ AnomalyGPT: Detecting Industrial Anomalies using Large Vision-Language Models [[AAAI 2024]](https://ojs.aaai.org/index.php/AAAI/article/view/27963)[[code]](https://github.com/CASIA-IVA-Lab/AnomalyGPT)[[project page]](https://anomalygpt.github.io/)
+ A Comprehensive Augmentation Framework for Anomaly Detection [[AAAI 2024]](https://ojs.aaai.org/index.php/AAAI/article/view/28720)


## WACV 2024
+ ReConPatch: Contrastive Patch Representation Learning for Industrial Anomaly Detection [[WACV 2024]](https://openaccess.thecvf.com/content/WACV2024/papers/Hyun_ReConPatch_Contrastive_Patch_Representation_Learning_for_Industrial_Anomaly_Detection_WACV_2024_paper.pdf)
+ Learning Transferable Representations for Image Anomaly Localization Using Dense Pretraining [[WACV 2024]](https://openaccess.thecvf.com/content/WACV2024/papers/He_Learning_Transferable_Representations_for_Image_Anomaly_Localization_Using_Dense_Pretraining_WACV_2024_paper.pdf)[[code]](https://github.com/terrlo/DS2)
+ EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies [[WACV 2024]](https://openaccess.thecvf.com/content/WACV2024/papers/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.pdf)
+ Contextual Affinity Distillation for Image Anomaly Detection [[WACV 2024]](https://openaccess.thecvf.com/content/WACV2024/papers/Zhang_Contextual_Affinity_Distillation_for_Image_Anomaly_Detection_WACV_2024_paper.pdf)
+ Attention Modules Improve Image-Level Anomaly Detection for Industrial Inspection: A DifferNet Case Study [[WACV 2024]](https://openaccess.thecvf.com/content/WACV2024/papers/Vieira_e_Silva_Attention_Modules_Improve_Image-Level_Anomaly_Detection_for_Industrial_Inspection_A_WACV_2024_paper.pdf)
+ PromptAD: Zero-shot Anomaly Detection using Text Prompts [[WACV 2024]](https://openaccess.thecvf.com/content/WACV2024/papers/Li_PromptAD_Zero-Shot_Anomaly_Detection_Using_Text_Prompts_WACV_2024_paper.pdf)
+ High-Fidelity Zero-Shot Texture Anomaly Localization Using Feature Correspondence Analysis [[WACV 2024]](https://openaccess.thecvf.com/content/WACV2024/html/Ardelean_High-Fidelity_Zero-Shot_Texture_Anomaly_Localization_Using_Feature_Correspondence_Analysis_WACV_2024_paper.html)
+ Cheating Depth: Enhancing 3D Surface Anomaly Detection via Depth Simulation [[WACV 2024]](https://openaccess.thecvf.com/content/WACV2024/papers/Zavrtanik_Cheating_Depth_Enhancing_3D_Surface_Anomaly_Detection_via_Depth_Simulation_WACV_2024_paper.pdf)[[code]](https://github.com/VitjanZ/3DSR)

## NeurIPS 2023
+ Real3D-AD: A Dataset of Point Cloud Anomaly Detection [[NeurIPS 2023]](https://openreview.net/pdf?id=zGthDp4yYe)[[code]](https://github.com/M-3LAB/Real3D-AD)[[中文]](https://blog.csdn.net/m0_63828250/article/details/136667168)
+ PAD: A Dataset and Benchmark for Pose-agnostic Anomaly Detection [[NeurIPS 2023]](https://openreview.net/pdf?id=kxFKgqwFNk)[[code]](https://github.com/EricLee0224/PAD)
+ Zero-Shot Anomaly Detection via Batch Normalization [[NeurIPS 2023]](https://openreview.net/pdf?id=d1wjMBYbP1)[[code]](https://github.com/aodongli/zero-shot-ad-via-batch-norm)
+ SANFlow: Semantic-Aware Normalizing Flow for Anomaly Detection and Localization [[NeurIPS 2023]](https://openreview.net/pdf?id=BqZ70BEtuW)
+ Energy-Based Models for Anomaly Detection: A Manifold Diffusion Recovery Approach [[NeurIPS 2023]](https://openreview.net/pdf?id=4nSDDokpfK)
+ Hierarchical Vector Quantized Transformer for Multi-class Unsupervised Anomaly Detection [[NeurIPS 2023]](https://openreview.net/pdf?id=clJTNssgn6)[[code]](https://github.com/RuiyingLu/HVQ-Trans)
+ ReContrast: Domain-Specific Anomaly Detection via Contrastive Reconstruction [[NeurIPS 2023]](https://openreview.net/pdf?id=KYxD9YCQBH)[[code]](https://github.com/guojiajeremy/ReContrast)

<!-- 
## ICML 2023
+ Shape-Guided Dual-Memory Learning for 3D Anomaly Detection [[ICML 2023]](https://openreview.net/forum?id=IkSGn9fcPz)
+ Fascinating Supervisory Signals and Where to Find Them: Deep Anomaly Detection with Scale Learning [[ICML 2023]](https://openreview.net/forum?id=V6PNBRWRil)

## ACM MM 2023
+ EasyNet: An Easy Network for 3D Industrial Anomaly Detection [[ACM MM 2023]](https://arxiv.org/abs/2307.13925)

## ICCV 2023
+ Remembering Normality: Memory-guided Knowledge Distillation for Unsupervised Anomaly Detection [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Gu_Remembering_Normality_Memory-guided_Knowledge_Distillation_for_Unsupervised_Anomaly_Detection_ICCV_2023_paper.pdf)
+ Unsupervised Surface Anomaly Detection with Diffusion Probabilistic Model [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Unsupervised_Surface_Anomaly_Detection_with_Diffusion_Probabilistic_Model_ICCV_2023_paper.pdf)
+ PNI: Industrial Anomaly Detection using Position and Neighborhood Information [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Bae_PNI__Industrial_Anomaly_Detection_using_Position_and_Neighborhood_Information_ICCV_2023_paper.pdf)[[code]](https://github.com/wogur110/PNI_Anomaly_Detection)
+ Anomaly Detection using Score-based Perturbation Resilience [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Shin_Anomaly_Detection_using_Score-based_Perturbation_Resilience_ICCV_2023_paper.pdf)
+ Template-guided Hierarchical Feature Restoration for Anomaly Detection [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Guo_Template-guided_Hierarchical_Feature_Restoration_for_Anomaly_Detection_ICCV_2023_paper.pdf)
+ Focus the Discrepancy: Intra- and Inter-Correlation Learning for Image Anomaly Detection [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Yao_Focus_the_Discrepancy_Intra-_and_Inter-Correlation_Learning_for_Image_Anomaly_ICCV_2023_paper.pdf)[[code]](https://github.com/xcyao00/FOD)
+ Anomaly Detection under Distribution Shift [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Cao_Anomaly_Detection_Under_Distribution_Shift_ICCV_2023_paper.pdf)[[code]](https://github.com/mala-lab/ADShift)
+ FastRecon: Few-shot Industrial Anomaly Detection via Fast Feature Reconstruction [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Fang_FastRecon_Few-shot_Industrial_Anomaly_Detection_via_Fast_Feature_Reconstruction_ICCV_2023_paper.pdf)[[code]](https://github.com/FzJun26th/FastRecon)
+ Inter-Realization Channels: Unsupervised Anomaly Detection Beyond One-Class Classification [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/McIntosh_Inter-Realization_Channels_Unsupervised_Anomaly_Detection_Beyond_One-Class_Classification_ICCV_2023_paper.pdf)[[code]](https://github.com/DeclanMcIntosh/InReaCh)
+ Removing Anomalies as Noises for Industrial Defect Localization [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Lu_Removing_Anomalies_as_Noises_for_Industrial_Defect_Localization_ICCV_2023_paper.pdf)

-->

## LLM related
+ Myriad: Large Multimodal Model by Applying Vision Experts for Industrial Anomaly Detection [[2023]](https://arxiv.org/abs/2310.19070)[[code]](https://github.com/tzjtatata/Myriad)
+ AnomalyGPT: Detecting Industrial Anomalies using Large Vision-Language Models [[AAAI 2024]](https://arxiv.org/abs/2308.15366)[[code]](https://github.com/CASIA-IVA-Lab/AnomalyGPT)[[project page]](https://anomalygpt.github.io/)
+ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision) [[2023 Section 9.2]](https://arxiv.org/abs/2309.17421)
+ Towards Generic Anomaly Detection and Understanding: Large-scale Visual-linguistic Model (GPT-4V) Takes the Lead [[2023]](https://arxiv.org/abs/2311.02782)[[code]](https://github.com/caoyunkang/GPT4V-for-Generic-Anomaly-Detection)
+ Exploring Grounding Potential of VQA-oriented GPT-4V for Zero-shot Anomaly Detection [[2023]](https://arxiv.org/abs/2311.02612)[[code]](https://github.com/zhangzjn/GPT-4V-AD)
+ Customizing Visual-Language Foundation Models for Multi-modal Anomaly Detection and Reasoning [[2024]](https://arxiv.org/abs/2403.11083)
+ Do LLMs Understand Visual Anomalies? Uncovering LLM Capabilities in Zero-shot Anomaly Detection [[2024]](https://arxiv.org/abs/2404.09654)
+ LogiCode: an LLM-Driven Framework for Logical Anomaly Detection [[2024]](https://arxiv.org/pdf/2406.04687)
+ FabGPT: An Efficient Large Multimodal Model for Complex Wafer Defect Knowledge Queries [[ICCAD 2024]](https://arxiv.org/abs/2407.10810)
+ VMAD: Visual-enhanced Multimodal Large Language Model for Zero-Shot Anomaly Detection [[2024]](https://arxiv.org/abs/2409.20146)
+ Are Anomaly Scores Telling the Whole Story? A Benchmark for Multilevel Anomaly Detection [[2024]](https://arxiv.org/abs/2411.14515)
<!--
## CVPR 2023
+ CVPR 2023 Tutorial on "Recent Advances in Anomaly Detection" [[CVPR Workshop 2023(mainly on video anomaly detection)]](https://sites.google.com/view/cvpr2023-tutorial-on-ad/)[[video]](https://www.youtube.com/watch?v=dXxrzWeybBo&feature=youtu.be)
+ Workshop on Vision-Based Industrial Inspection [[CVPR Workshop paper list 2023]](https://openaccess.thecvf.com/CVPR2023_workshops/VISION)
+ Visual Anomaly and Novelty Detection [[CVPR Workshop paper list 2023]](https://openaccess.thecvf.com/CVPR2023_workshops/VAND)
+ Revisiting Reverse Distillation for Anomaly Detection [[CVPR 2023]](https://openaccess.thecvf.com/content/CVPR2023/papers/Tien_Revisiting_Reverse_Distillation_for_Anomaly_Detection_CVPR_2023_paper.pdf) [[code]](https://github.com/tientrandinh/Revisiting-Reverse-Distillation)
+ OmniAL A unifiled CNN framework for unsupervised anomaly localization [[CVPR 2023]](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_OmniAL_A_Unified_CNN_Framework_for_Unsupervised_Anomaly_Localization_CVPR_2023_paper.pdf)
+ Explicit Boundary Guided Semi-Push-Pull Contrastive Learning for Supervised Anomaly Detection [[CVPR 2023]](https://arxiv.org/abs/2207.01463)[[code]](https://github.com/xcyao00/BGAD)
+ DeSTSeg: Segmentation Guided Denoising Student-Teacher for Anomaly Detection [[CVPR 2023]](https://arxiv.org/abs/2211.11317)[[code]](https://github.com/apple/ml-destseg)
+ Diversity-Measurable Anomaly Detection [[CVPR 2023]](https://arxiv.org/abs/2303.05047)
+ WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation [[CVPR 2023]](https://arxiv.org/abs/2303.14814)
+ SimpleNet: A Simple Network for Image Anomaly Detection and Localization [[CVPR 2023]](https://arxiv.org/abs/2303.15140)[[code]](https://github.com/DonaldRR/SimpleNet)
+ PyramidFlow: High-Resolution Defect Contrastive Localization using Pyramid Normalizing Flow [[CVPR 2023]](https://arxiv.org/abs/2303.02595)[[code]](https://github.com/gasharper/PyramidFlow)
+ Multimodal Industrial Anomaly Detection via Hybrid Fusion [[CVPR 2023]](https://arxiv.org/abs/2303.00601)[[code]](https://github.com/nomewang/M3DM)
+ Prototypical Residual Networks for Anomaly Detection and Localization [[CVPR 2023]](https://arxiv.org/abs/2212.02031)[[code]](https://github.com/xcyao00/PRNet)
+ SQUID: Deep Feature In-Painting for Unsupervised Anomaly Detection [[CVPR 2023]](https://arxiv.org/abs/2111.13495)
+ APRIL-GAN: A Zero-/Few-Shot Anomaly Classification and Segmentation Method for CVPR 2023 VAND Workshop Challenge Tracks 1&2: 1st Place on Zero-shot AD and 4th Place on Few-shot AD [[CVPR 2023 VAND Workshop Challenge]](https://arxiv.org/abs/2305.17382)
-->

## SAM segment anything
+ Segment Anything Is Not Always Perfect: An Investigation of SAM on Different Real-world Applications [[2023 SAM tech report]](https://arxiv.org/abs/2304.05750)
+ SAM Struggles in Concealed Scenes -- Empirical Study on "Segment Anything" [[2023 SAM tech report]](https://arxiv.org/abs/2304.06022)
+ Segment Any Anomaly without Training via Hybrid Prompt Regularization [[2023]](https://arxiv.org/abs/2305.10724) [[code]](https://github.com/caoyunkang/GroundedSAM-zero-shot-anomaly-detection)
+ Application of Segment Anything Model for Civil Infrastructure Defect Assessment [[2023 SAM tech report]](https://arxiv.org/abs/2304.12600)
+ Segment Anything in Defect Detection [[2023]](https://arxiv.org/abs/2311.10245)
+ Unsupervised Continual Anomaly Detection with Contrastively-learned Prompt [[AAAI 2024]](https://ojs.aaai.org/index.php/AAAI/article/view/28153)[[code]](https://github.com/shirowalker/UCAD)
+ ClipSAM: CLIP and SAM Collaboration for Zero-Shot Anomaly Segmentation [[2023]](https://arxiv.org/pdf/2401.12665)
+ A SAM-guided Two-stream Lightweight Model for Anomaly Detection [[2024]](https://arxiv.org/abs/2402.19145)[[code]](https://github.com/StitchKoala/STLM)
+ Inspiring the Next Generation of Segment Anything Models: Comprehensively Evaluate SAM and SAM 2 with Diverse Prompts Towards Context-Dependent Concepts under Different Scenes [[2024]](https://arxiv.org/abs/2412.01240)[[code]](https://github.com/lartpang/SAMs-CDConcepts-Eval)

<!--
## ICLR 2023
+ Pushing the Limits of Fewshot Anomaly Detection in Industry Vision: Graphcore [[ICLR 2023]](https://openreview.net/pdf?id=xzmqxHdZAwO)
+ RGI: robust GAN-inversion for mask-free image inpainting and unsupervised pixel-wise anomaly detection [[ICLR 2023]](https://openreview.net/pdf?id=1UbNwQC89a)
-->

## Others
+ Self-supervised Context Learning for Visual Inspection of Industrial Defects [[2023]](https://arxiv.org/abs/2311.06504)[[code]](https://github.com/wangpeng000/VisualInspection)
+ CLIP-AD: A Language-Guided Staged Dual-Path Model for Zero-shot Anomaly Detection [[2023]](https://arxiv.org/abs/2311.00453)
+ Self-Tuning Self-Supervised Anomaly Detection [[2023]](https://openreview.net/forum?id=saj54kqrBj)
+ Model Selection of Anomaly Detectors in the Absence of Labeled Validation Data [[2023]](https://arxiv.org/abs/2310.10461)
+ A Discrepancy Aware Framework for Robust Anomaly Detection [[2023]](https://arxiv.org/abs/2310.07585)[[code]](https://github.com/caiyuxuan1120/DAF)
+ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision) [[2023 Section 9.2]](https://arxiv.org/abs/2309.17421)
+ Global Context Aggregation Network for Lightweight Saliency Detection of Surface Defects [[2023]](https://arxiv.org/abs/2309.12641)
+ Decision Fusion Network with Perception Fine-tuning for Defect Classification [[2023]](https://arxiv.org/abs/2309.12630)
+ FAIR: Frequency-aware Image Restoration for Industrial Visual Anomaly Detection [[2023]](https://arxiv.org/abs/2309.07068)[[code]](https://github.com/liutongkun/FAIR)
+ AnoVL: Adapting Vision-Language Models for Unified Zero-shot Anomaly Localization [[2023]](https://arxiv.org/abs/2308.15939)[[code]](https://github.com/hq-deng/AnoVL)
+ End-to-End Augmentation Hyperparameter Tuning for Self-Supervised Anomaly Detection [[2023]](https://arxiv.org/abs/2306.12033)
+ CVPR 1st workshop on Vision-based InduStrial InspectiON [[CVPR 2023 Workshop]](https://vision-based-industrial-inspection.github.io/cvpr-2023/) [[data link]](https://drive.google.com/drive/folders/1TVp_UXJuXudqhC2L3ZKyIDcmQ_2O3JVi)
+ Multilevel Saliency-Guided Self-Supervised Learning for Image Anomaly Detection [[2023]](http://arxiv.org/pdf/2311.18332v1)
+ How Low Can You Go? Surfacing Prototypical In-Distribution Samples for Unsupervised Anomaly Detection [Dataset Distillation][[2023]](http://arxiv.org/pdf/2312.03804v1)
+ Exploring Plain ViT Reconstruction for Multi-class Unsupervised Anomaly Detection [[2023]](https://arxiv.org/abs/2312.07495)
+ AUPIMO: Redefining Visual Anomaly Detection Benchmarks with High Speed and Low Tolerance [[2024]](https://arxiv.org/abs/2401.01984)
+ Model Selection of Zero-shot Anomaly Detectors in the Absence of Labeled Validation Data [[2024]](https://arxiv.org/abs/2310.10461)
+ PUAD: Frustratingly Simple Method for Robust Anomaly Detection [[2024]](https://arxiv.org/abs/2402.15143)
+ COFT-AD: COntrastive Fine-Tuning for Few-Shot Anomaly Detection [[TIP2024]](http://arxiv.org/abs/2402.18998)
+ PointCore: Efficient Unsupervised Point Cloud Anomaly Detector Using Local-Global Features [[2024]](https://arxiv.org/abs/2403.01804)
+ Learning Unified Reference Representation for Unsupervised Multi-class Anomaly Detection [[2024]](https://arxiv.org/abs/2403.11561)
+ RAD: A Comprehensive Dataset for Benchmarking the Robustness of Image Anomaly Detection [[CASE 2024]](https://arxiv.org/abs/2406.07176)[[github page]](https://github.com/hustCYQ/RAD-dataset)
+ Towards Zero-shot Point Cloud Anomaly Detection: A Multi-View Projection Framework [[2024]](https://arxiv.org/abs/2409.13162)[[code]](https://github.com/hustCYQ/MVP-PCLIP)

## Medical (related)
+ Towards Universal Unsupervised Anomaly Detection in Medical Imaging [[2024]](http://arxiv.org/pdf/2401.10637v1)
+ MAEDiff: Masked Autoencoder-enhanced Diffusion Models for Unsupervised Anomaly Detection in Brain Images [[2024]](http://arxiv.org/pdf/2401.10561v1)
+ BMAD: Benchmarks for Medical Anomaly Detection [[2023]](https://arxiv.org/abs/2306.11876)
+ Unsupervised Pathology Detection: A Deep Dive Into the State of the Art [[2023]](https://arxiv.org/abs/2303.00609)
+ Adapting Visual-Language Models for Generalizable Anomaly Detection in Medical Images [[CVPR 2024]](https://arxiv.org/abs/2403.12570)
+ Multi-Image Visual Question Answering for Unsupervised Anomaly Detection [[2024]](http://arxiv.org/abs/2404.07622v1)

# Paper Tree (Classification of representative methods)
![PaperTree](https://github.com/M-3LAB/awesome-industrial-anomaly-detection/blob/main/paper_tree.png)
# Timeline
![Timeline](https://github.com/M-3LAB/awesome-industrial-anomaly-detection/blob/main/timeline.png)

# Paper list for industrial image anomaly detection

# Related Survey, Benchmark, and Framework
+ A review on computer vision based defect detection and condition assessment of concrete and asphalt civil infrastructure [[2015]](https://www.sciencedirect.com/science/article/abs/pii/S1474034615000208)
+ Visual-based defect detection and classification approaches for industrial applications: a survey [[2020]](https://pdfs.semanticscholar.org/1dfc/080a5f26b5ce78f9ce3e9f106bf7e8124f74.pdf)
+ A Unified Survey on Anomaly, Novelty, Open-Set, and Out-of-Distribution Detection: Solutions and Future Challenges [[TMLR 2022]](https://arxiv.org/abs/2110.14051)
+ Deep Learning for Unsupervised Anomaly Localization in Industrial Images: A Survey [[TIM 2022]](http://arxiv.org/pdf/2207.10298)
+ A Survey on Unsupervised Industrial Anomaly Detection Algorithms [[2022]](https://arxiv.org/abs/2204.11161)
+ A Survey of Methods for Automated Quality Control Based on Images [[IJCV 2023]](https://link.springer.com/article/10.1007/s11263-023-01822-w)[[github page]](https://github.com/jandiers/mvtec-results)
+ Benchmarking Unsupervised Anomaly Detection and Localization [[2022]](https://arxiv.org/abs/2205.14852)
+ IM-IAD: Industrial Image Anomaly Detection Benchmark in Manufacturing [[TCYB 2024]](https://arxiv.org/abs/2301.13359)[[code]](https://github.com/M-3LAB/open-iad)[[中文]](https://blog.csdn.net/m0_63828250/article/details/136891730)
+ A Deep Learning-based Software for Manufacturing Defect Inspection [[TII 2017]](https://ieeexplore.ieee.org/document/9795891)[[code]](https://github.com/sundyCoder/DEye)
+ Anomalib: A Deep Learning Library for Anomaly Detection [[ICIP 2022]](https://ieeexplore.ieee.org/abstract/document/9897283/)[[code]](https://github.com/openvinotoolkit/anomalib)
+ Ph.D. thesis of Paul Bergmann(The first author of MVTec AD series) [[2022]](https://mediatum.ub.tum.de/1662158)
+ CVPR 2023 Tutorial on "Recent Advances in Anomaly Detection" [[CVPR Workshop 2023]](https://sites.google.com/view/cvpr2023-tutorial-on-ad/)[[video]](https://www.youtube.com/watch?v=dXxrzWeybBo&feature=youtu.be)
+ Exploring Plain ViT Reconstruction for Multi-class Unsupervised Anomaly Detection [[2023]](https://arxiv.org/abs/2312.07495)[[code]](https://github.com/zhangzjn/ADer)
+ A Survey on Visual Anomaly Detection: Challenge, Approach, and Prospect [[2024]](https://arxiv.org/pdf/2401.16402.pdf)
+ AUPIMO: Redefining Visual Anomaly Detection Benchmarks with High Speed and Low Tolerance [[2024]](https://arxiv.org/abs/2401.01984)
+ Explainable Anomaly Detection in Images and Videos: A Survey [[2024]](https://arxiv.org/pdf/2302.06670)[[repo]](https://github.com/wyzjack/Awesome-XAD)
+ RAD: A Comprehensive Dataset for Benchmarking the Robustness of Image Anomaly Detection [[CASE 2024]](https://arxiv.org/abs/2406.07176)[[github page]](https://github.com/hustCYQ/RAD-dataset)
+ Generalized Out-of-Distribution Detection and Beyond in Vision Language Model Era: A Survey [[2024]](https://arxiv.org/abs/2407.21794)[[github page]](https://github.com/AtsuMiyai/Awesome-OOD-VLM)
+ Large Language Models for Anomaly and Out-of-Distribution Detection: A Survey [[2024]](https://arxiv.org/abs/2409.01980)[[github page]](https://github.com/rux001/Awesome-LLM-Anomaly-OOD-Detection)
+ A Survey on RGB, 3D, and Multimodal Approaches for Unsupervised Industrial Anomaly Detection [[2024]](https://arxiv.org/abs/2410.21982)[[github page]](https://github.com/Sunny5250/Awesome-Multi-Setting-UIAD)
+ OpenOOD: Benchmarking Generalized Out-of-Distribution Detection [[NeurIPS2022v1]](https://openreview.net/pdf?id=gT6j4_tskUt)[[2024v1.5]](https://arxiv.org/abs/2306.09301)[[github page]](https://github.com/Jingkang50/OpenOOD)

# 2 Unsupervised AD

## 2.1 Feature-Embedding-based Methods

### 2.1.1 Teacher-Student
+ Contextual Affinity Distillation for Image Anomaly Detection [[WACV 2024]](https://openaccess.thecvf.com/content/WACV2024/papers/Zhang_Contextual_Affinity_Distillation_for_Image_Anomaly_Detection_WACV_2024_paper.pdf)
+ Revisiting Reverse Distillation for Anomaly Detection [[CVPR 2023]](https://openaccess.thecvf.com/content/CVPR2023/papers/Tien_Revisiting_Reverse_Distillation_for_Anomaly_Detection_CVPR_2023_paper.pdf) [[code]](https://github.com/tientrandinh/Revisiting-Reverse-Distillation)
+ Uninformed students: Student-teacher anomaly detection with discriminative latent embeddings [[CVPR 2020]](http://arxiv.org/pdf/1911.02357)
+ Multiresolution knowledge distillation for anomaly detection [[CVPR 2021]](https://arxiv.org/pdf/2011.11108)
+ Glancing at the Patch: Anomaly Localization With Global and Local Feature Comparison [[CVPR 2021]](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Glancing_at_the_Patch_Anomaly_Localization_With_Global_and_Local_CVPR_2021_paper.html)
+ Reconstruction Student with Attention for Student-Teacher Pyramid Matching [[2021]](https://arxiv.org/pdf/2111.15376.pdf)
+ Student-Teacher Feature Pyramid Matching for Anomaly Detection [[2021]](https://arxiv.org/pdf/2103.04257.pdf)[[code]](https://github.com/smiler96/PFM-and-PEFM-for-Image-Anomaly-Detection-and-Segmentation)
+ PFM and PEFM for Image Anomaly Detection and Segmentation [[CASE 2022]](https://ieeexplore.ieee.org/abstract/document/9926547/) [[TII 2022]](https://ieeexplore.ieee.org/document/9795121)[[code]](https://github.com/smiler96/PFM-and-PEFM-for-Image-Anomaly-Detection-and-Segmentation)
+ Reconstructed Student-Teacher and Discriminative Networks for Anomaly Detection [[2022]](https://arxiv.org/pdf/2210.07548.pdf)
+ Anomaly Detection via Reverse Distillation from One-Class Embedding [[CVPR 2022]](http://arxiv.org/pdf/2201.10703)[[code]](https://github.com/hq-deng/RD4AD)
+ Asymmetric Student-Teacher Networks for Industrial Anomaly Detection [[WACV 2022]](https://arxiv.org/pdf/2210.07829.pdf)[[code]](https://github.com/marco-rudolph/AST)
+ Informative knowledge distillation for image anomaly segmentation [[2022]](https://www.sciencedirect.com/science/article/pii/S0950705122004038/pdfft?md5=758c327dd4d1d052b61a19882f957123&pid=1-s2.0-S0950705122004038-main.pdf)[[code]](https://github.com/caoyunkang/IKD)
+ Remembering Normality: Memory-guided Knowledge Distillation for Unsupervised Anomaly Detection [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Gu_Remembering_Normality_Memory-guided_Knowledge_Distillation_for_Unsupervised_Anomaly_Detection_ICCV_2023_paper.pdf)
+ A Discrepancy Aware Framework for Robust Anomaly Detection [[2023]](https://arxiv.org/abs/2310.07585)[[code]](https://github.com/caiyuxuan1120/DAF)
+ Enhanced multi-scale features mutual mapping fusion based on reverse knowledge distillation for industrial anomaly detection and localization [[TBD 2024]](https://ieeexplore.ieee.org/abstract/document/10382612)
+ AEKD: Unsupervised auto-encoder knowledge distillation for industrial anomaly detection [[JMS 2024]](https://www.sciencedirect.com/science/article/pii/S0278612524000244)
+ Masked feature regeneration based asymmetric student–teacher network for anomaly detection [[Multimedia Tools and Applications 2024]](https://link.springer.com/article/10.1007/s11042-024-18512-5)
+ Feature-Constrained and Attention-Conditioned Distillation Learning for Visual Anomaly Detection [[ICASSP 2024]](https://ieeexplore.ieee.org/document/10448432)
+ MiniMaxAD: A Lightweight Autoencoder for Feature-Rich Anomaly Detection [[2024]](https://arxiv.org/abs/2405.09933)

### 2.1.2 One-Class Classification (OCC)
+ Patch svdd: Patch-level svdd for anomaly detection and segmentation [[ACCV 2020]](https://arxiv.org/pdf/2006.16067.pdf)
+ Anomaly detection using improved deep SVDD model with data structure preservation [[2021]](https://www.sciencedirect.com/science/article/am/pii/S0167865521001598)
+ A Semantic-Enhanced Method Based On Deep SVDD for Pixel-Wise Anomaly Detection [[2021]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9428370)
+ MOCCA: Multilayer One-Class Classification for Anomaly Detection [[2021]](http://arxiv.org/pdf/2012.12111)
+ Defect Detection of Metal Nuts Applying Convolutional Neural Networks [[2021]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9529439)
+ Panda: Adapting pretrained features for anomaly detection and segmentation [[2021]](http://arxiv.org/pdf/2010.05903)
+ Mean-shifted contrastive loss for anomaly detection [[2021]](https://arxiv.org/pdf/2106.03844.pdf)
+ Learning and Evaluating Representations for Deep One-Class Classification [[2020]](https://arxiv.org/pdf/2011.02578.pdf)
+ Self-supervised learning for anomaly detection with dynamic local augmentation [[2021]](https://ieeexplore.ieee.org/ielx7/6287639/6514899/09597511.pdf)
+ Contrastive Predictive Coding for Anomaly Detection [[2021]](https://arxiv.org/pdf/2107.07820.pdf)
+ Cutpaste: Self-supervised learning for anomaly detection and localization [[ICCV 2021]](http://arxiv.org/pdf/2104.04015)[[unofficial code]](https://github.com/Runinho/pytorch-cutpaste)
+ Consistent estimation of the max-flow problem: Towards unsupervised image segmentation [[2020]](http://arxiv.org/pdf/1811.00220)
+ MemSeg: A semi-supervised method for image surface defect detection using differences and commonalities [[2022]](https://arxiv.org/pdf/2205.00908.pdf)[[unofficial code]](https://github.com/TooTouch/MemSeg)
+ SimpleNet: A Simple Network for Image Anomaly Detection and Localization [[CVPR 2023]](https://github.com/DonaldRR/SimpleNet)[[code]](https://github.com/DonaldRR/SimpleNet)
+ End-to-End Augmentation Hyperparameter Tuning for Self-Supervised Anomaly Detection [[2023]](https://arxiv.org/abs/2306.12033)
+ Anomaly Detection under Distribution Shift [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Cao_Anomaly_Detection_Under_Distribution_Shift_ICCV_2023_paper.pdf)[[code]](https://github.com/mala-lab/ADShift)
+ Learning Transferable Representations for Image Anomaly Localization Using Dense Pretraining [[WACV 2024]](https://openaccess.thecvf.com/content/WACV2024/papers/He_Learning_Transferable_Representations_for_Image_Anomaly_Localization_Using_Dense_Pretraining_WACV_2024_paper.pdf)[[code]](https://github.com/terrlo/DS2)
+ GeneralAD: Anomaly Detection Across Domains by Attending to Distorted Features [[ECCV 2024]](https://arxiv.org/abs/2407.12427)[[code]](https://github.com/LucStrater/GeneralAD)
+ A Unified Anomaly Synthesis Strategy with Gradient Ascent for Industrial Anomaly Detection and Localization [[ECCV 2024]](https://arxiv.org/abs/2407.09359)[[code]](https://github.com/cqylunlun/GLASS)
+ Dual-Modeling Decouple Distillation for Unsupervised Anomaly Detection [[ACM MM 2024]](https://arxiv.org/abs/2408.03888)
+ SuperSimpleNet: Unifying Unsupervised and Supervised Learning for Fast and Reliable Surface Defect Detection [[ICIP 2024]](https://arxiv.org/abs/2408.03143)[[code]](https://github.com/blaz-r/SuperSimpleNet/tree/main)
+ Progressive Boundary Guided Anomaly Synthesis for Industrial Anomaly Detection [[TCSVT 2024]](https://ieeexplore.ieee.org/document/10716437)[[code]](https://github.com/cqylunlun/PBAS)

### 2.1.3 Distribution-Map
+ Anomaly Detection in Nanofibrous Materials by CNN-Based Self-Similarity [[Sensors 2018]](https://www.mdpi.com/1424-8220/18/1/209)
+ A Multi-Scale A Contrario method for Unsupervised Image Anomaly Detection [[2021]](http://arxiv.org/pdf/2110.02407)
+ Modeling the distribution of normal data in pre-trained deep features for anomaly detection [[2021]](http://arxiv.org/pdf/2005.14140)
+ Transfer Learning Gaussian Anomaly Detection by Fine-Tuning Representations [[2021]](https://arxiv.org/pdf/2108.04116.pdf)
+ PEDENet: Image anomaly localization via patch embedding and density estimation [[2022]](https://arxiv.org/pdf/2110.15525.pdf)
+ Unsupervised image anomaly detection and segmentation based on pre-trained feature mapping [[2022]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9795121)
+ Position Encoding Enhanced Feature Mapping for Image Anomaly Detection [[2022]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9926547)[[code]](https://github.com/smiler96/PFM-and-PEFM-for-Image-Anomaly-Detection-and-Segmentation)
+ Focus your distribution: Coarse-to-fine non-contrastive learning for anomaly detection and localization [[ICME 2022]](http://arxiv.org/pdf/2110.04538)
+ Anomaly Detection of Defect using Energy of Point Pattern Features within Random Finite Set Framework [[2021]](https://arxiv.org/abs/2108.12159)[[code]](https://github.com/AmmarKamoona/RFS-Energy-Anomaly-Detection-of-Defect)
+ Fastflow: Unsupervised anomaly detection and localization via 2d normalizing flows [[2021]](https://arxiv.org/pdf/2111.07677.pdf)[[unofficial code]](https://github.com/gathierry/FastFlow)
+ Same same but differnet: Semi-supervised defect detection with normalizing flows [[WACV 2021]](http://arxiv.org/pdf/2008.12577)[[code]](https://github.com/marco-rudolph/differnet)
+ Fully convolutional cross-scale-flows for image-based defect detection [[WACV 2022]](http://arxiv.org/pdf/2110.02855)[[code]](https://github.com/marco-rudolph/cs-flow)
+ Cflow-ad: Real-time unsupervised anomaly detection with localization via conditional normalizing flows [[WACV 2022]](http://arxiv.org/pdf/2107.12571)[[code]](https://github.com/gudovskiy/cflow-ad)
+ CAINNFlow: Convolutional block Attention modules and Invertible Neural Networks Flow for anomaly detection and localization tasks [[2022]](https://arxiv.org/pdf/2206.01992.pdf)
+ AltUB: Alternating Training Method to Update Base Distribution of Normalizing Flow for Anomaly Detection [[2022]](https://arxiv.org/pdf/2210.14913.pdf)
+ Collaborative Discrepancy Optimization for Reliable Image Anomaly Localization [[TII 2023]](https://ieeexplore.ieee.org/document/10034849)[[code]](https://github.com/caoyunkang/CDO)
+ PyramidFlow: High-Resolution Defect Contrastive Localization using Pyramid Normalizing Flow [[CVPR 2023]](https://arxiv.org/abs/2303.02595)[[code]](https://github.com/gasharper/PyramidFlow)
+ Attention Modules Improve Image-Level Anomaly Detection for Industrial Inspection: A DifferNet Case Study [[WACV 2024]](https://openaccess.thecvf.com/content/WACV2024/papers/Vieira_e_Silva_Attention_Modules_Improve_Image-Level_Anomaly_Detection_for_Industrial_Inspection_A_WACV_2024_paper.pdf)
+ Fascinating Supervisory Signals and Where to Find Them: Deep Anomaly Detection with Scale Learning [[ICML 2023]](https://openreview.net/forum?id=V6PNBRWRil)
+ FRAnomaly: flow-based rapid anomaly detection from images [[Applied Intelligence 2024]](https://link.springer.com/article/10.1007/s10489-024-05332-1)
+ Image alignment-based patch distribution framework for anomaly detection [[ICCVDM 2024]](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13063/130630O/Image-alignment-based-patch-distribution-framework-for-anomaly-detection/10.1117/12.3021499.full)
+ MSFlow: Multi-Scale Flow-based Framework for Unsupervised Anomaly Detection [[2024]](https://arxiv.org/abs/2308.15300)[[code]](https://github.com/cool-xuan/msflow)

### 2.1.4 Memory Bank
 + ReConPatch: Contrastive Patch Representation Learning for Industrial Anomaly Detection [[WACV 2024]](https://openaccess.thecvf.com/content/WACV2024/papers/Hyun_ReConPatch_Contrastive_Patch_Representation_Learning_for_Industrial_Anomaly_Detection_WACV_2024_paper.pdf)
 + Sub-image anomaly detection with deep pyramid correspondences [[2020]](https://arxiv.org/pdf/2005.02357.pdf)
 + Semi-orthogonal embedding for efficient unsupervised anomaly segmentation [[2021]](https://arxiv.org/pdf/2105.14737.pdf)
 + Anomaly Detection Via Self-Organizing Map [[2021]](http://arxiv.org/pdf/2107.09903)
 + PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization [[ICPR 2021]](https://link.springer.com/chapter/10.1007/978-3-030-68799-1_35)[[unofficial code]](https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master)
 + Industrial Image Anomaly Localization Based on Gaussian Clustering of Pretrained Feature [[2021]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9479740)
 + Towards total recall in industrial anomaly detection[[CVPR 2022]](http://arxiv.org/pdf/2106.08265)[[code]](https://github.com/amazon-science/patchcore-inspection)
 + CFA: Coupled-Hypersphere-Based Feature Adaptation for Target-Oriented Anomaly Localization[[2022]](https://arxiv.org/pdf/2206.04325.pdf)[[code]](https://github.com/sungwool/CFA_for_anomaly_localization)
 + FAPM: Fast Adaptive Patch Memory for Real-time Industrial Anomaly Detection[[2022]](https://arxiv.org/pdf/2211.07381.pdf)
 + N-pad: Neighboring Pixel-based Industrial Anomaly Detection [[2022]](https://arxiv.org/pdf/2210.08768.pdf)
 + Multi-scale patch-based representation learning for image anomaly detection and segmentation [[2022]](https://openaccess.thecvf.com/content/WACV2022/papers/Tsai_Multi-Scale_Patch-Based_Representation_Learning_for_Image_Anomaly_Detection_and_Segmentation_WACV_2022_paper.pdf)
 + SPot-the-Difference Self-supervised Pre-training for Anomaly Detection and Segmentation [[ECCV 2022]](https://arxiv.org/pdf/2207.14315.pdf)
 + Diversity-Measurable Anomaly Detection [[CVPR 2023]](https://arxiv.org/abs/2303.05047)
 + SelFormaly: Towards Task-Agnostic Unified Anomaly Detection[[2023]](https://arxiv.org/abs/2307.12540)
 + REB: Reducing Biases in Representation for Industrial Anomaly Detection [[2023]](https://arxiv.org/abs/2308.12577)[[code]](https://github.com/ShuaiLYU/REB)
 + PNI : Industrial Anomaly Detection using Position and Neighborhood Information [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Bae_PNI__Industrial_Anomaly_Detection_using_Position_and_Neighborhood_Information_ICCV_2023_paper.pdf)[[code]](https://github.com/wogur110/PNI_Anomaly_Detection)
 + Inter-Realization Channels: Unsupervised Anomaly Detection Beyond One-Class Classification [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/McIntosh_Inter-Realization_Channels_Unsupervised_Anomaly_Detection_Beyond_One-Class_Classification_ICCV_2023_paper.pdf)[[code]](https://github.com/DeclanMcIntosh/InReaCh)
 + Grid-Based Continuous Normal Representation for Anomaly Detection [[2024]](https://arxiv.org/abs/2402.18293)[[code]](https://github.com/tae-mo/GRAD)
 + PointCore: Efficient Unsupervised Point Cloud Anomaly Detector Using Local-Global Features [[2024]](https://arxiv.org/abs/2403.01804)
 + DMAD: Dual Memory Bank for Real-World Anomaly Detection [[2024]](https://arxiv.org/abs/2403.12362)
 + A Reconstruction-Based Feature Adaptation for Anomaly Detection with Self-Supervised Multi-Scale Aggregation [[ICASSP 2024]](https://ieeexplore.ieee.org/document/10446766)
 + AnomalousPatchCore: Exploring the Use of Anomalous Samples in Industrial Anomaly Detection [[ECCVW 2024]](https://arxiv.org/abs/2408.15113)
 + VQ-Flow: Taming Normalizing Flows for Multi-Class Anomaly Detection via Hierarchical Vector Quantization [[2024]](https://arxiv.org/abs/2409.00942)[[code]](https://github.com/cool-xuan/vqflow)
 + FOCT: Few-shot Industrial Anomaly Detection with Foreground-aware Online Conditional Transport [[ACM MM 2024]](https://dl.acm.org/doi/10.1145/3664647.3680771)

 ### 2.1.5 Vison Language AD
 + Random Word Data Augmentation with CLIP for Zero-Shot Anomaly Detection [[BMVC 2023]](https://arxiv.org/abs/2308.11119)
 + AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection [[ICLR 2024]](https://openreview.net/forum?id=buC4E91xZE)[[code]](https://github.com/zqhang/AnomalyCLIP)
 + WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation [[CVPR 2023]](https://arxiv.org/abs/2303.14814)
 + ClipSAM: CLIP and SAM Collaboration for Zero-Shot Anomaly Segmentation [[2023]](https://arxiv.org/pdf/2401.12665)
 + CLIP-AD: A Language-Guided Staged Dual-Path Model for Zero-shot Anomaly Detection [[2023]](https://arxiv.org/abs/2311.00453)
 + AnoVL: Adapting Vision-Language Models for Unified Zero-shot Anomaly Localization [[2023]](https://arxiv.org/abs/2308.15939)[[code]](https://github.com/hq-deng/AnoVL)
 + AnomalyGPT: Detecting Industrial Anomalies using Large Vision-Language Models [[AAAI 2024]](https://arxiv.org/abs/2308.15366)[[code]](https://github.com/CASIA-IVA-Lab/AnomalyGPT)[[project page]](https://anomalygpt.github.io/)
 + Anomaly Detection by Adapting a pre-trained Vision Language Model [[2024]](https://arxiv.org/abs/2403.09493)
 + Customizing Visual-Language Foundation Models for Multi-modal Anomaly Detection and Reasoning [[2024]](https://arxiv.org/abs/2403.11083)[[code]](https://github.com/Xiaohao-Xu/Customizable-VLM)
 + PromptAD: Learning Prompts with only Normal Samples for Few-Shot Anomaly Detection [[CVPR 2024]](https://arxiv.org/abs/2404.05231)[[code]](https://github.com/FuNz-0/PromptAD)
 + Do LLMs Understand Visual Anomalies? Uncovering LLM Capabilities in Zero-shot Anomaly Detection [[2024]](https://arxiv.org/abs/2404.09654)
 + FiLo: Zero-Shot Anomaly Detection by Fine-Grained Description and High-Quality Localization [[2024]](https://arxiv.org/abs/2404.13671)
 + Dual-Image Enhanced CLIP for Zero-Shot Anomaly Detection [[2024]](https://arxiv.org/abs/2405.04782)
 + AnoPLe: Few-Shot Anomaly Detection via Bi-directional Prompt Learning with Only Normal Samples [[2024]](https://arxiv.org/abs/2408.13516)[[code]](https://github.com/YoojLee/AnoPLe)
 + GlocalCLIP: Object-agnostic Global-Local Prompt Learning for Zero-shot Anomaly Detection [[2024]](https://arxiv.org/abs/2411.06071)
 + UniVAD: A Training-free Unified Model for Few-shot Visual Anomaly Detection [[2024]](https://arxiv.org/abs/2412.03342)[[code]](https://uni-vad.github.io/#)

## 2.2 Reconstruction-Based Methods

### 2.2.1 Autoencoder (AE)
 + Improving unsupervised defect segmentation by applying structural similarity to autoencoders [[2018]](https://arxiv.org/pdf/1807.02011.pdf)
 + Automatic Fabric Defect Detection with a Multi-Scale Convolutional Denoising Autoencoder Network Model [[Sensors 2018]](https://www.mdpi.com/1424-8220/18/4/1064)
 + An Unsupervised-Learning-Based Approach for Automated Defect Inspection on Textured Surfaces [[TIM 2018]](https://ieeexplore.ieee.org/abstract/document/8281622)
 + Unsupervised anomaly detection using style distillation [[2020]](https://ieeexplore.ieee.org/ielx7/6287639/6514899/09288772.pdf)
 + Unsupervised two-stage anomaly detection [[2021]](https://arxiv.org/pdf/2103.11671.pdf)
 + Dfr: Deep feature reconstruction for unsupervised anomaly segmentation [[Neurocomputing 2020]](https://arxiv.org/pdf/2012.07122.pdf)
 + Unsupervised anomaly segmentation via multilevel image reconstruction and adaptive attention-level transition [[2021]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9521893)
 + Encoding structure-texture relation with p-net for anomaly detection in retinal images [[2020]](http://arxiv.org/pdf/2008.03632)
 + Improved anomaly detection by training an autoencoder with skip connections on images corrupted with stain-shaped noise [[2021]](http://arxiv.org/pdf/2008.12977)
 + Unsupervised anomaly detection for surface defects with dual-siamese network [[2022]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9681338)
 + Divide-and-assemble: Learning block-wise memory for unsupervised anomaly detection [[ICCV 2021]](https://openaccess.thecvf.com/content/ICCV2021/papers/Hou_Divide-and-Assemble_Learning_Block-Wise_Memory_for_Unsupervised_Anomaly_Detection_ICCV_2021_paper.pdf)
 + Reconstruction from edge image combined with color and gradient difference for industrial surface anomaly detection [[2022]](http://arxiv.org/pdf/2210.14485)[[code]](https://github.com/liutongkun/edgrec)
 + Spatial Contrastive Learning for Anomaly Detection and Localization [[2022]](https://ieeexplore.ieee.org/ielx7/6287639/9668973/09709224.pdf)
 + Superpixel masking and inpainting for self-supervised anomaly detection [[BMVC 2020]](https://www.bmvc2020-conference.com/assets/papers/0275.pdf)
 + Iterative image inpainting with structural similarity mask for anomaly detection [[2020]](https://openreview.net/pdf?id=b4ach0lGuYO)
 + Self-Supervised Masking for Unsupervised Anomaly Detection and Localization [[2022]](https://arxiv.org/pdf/2205.06568.pdf)
 + Reconstruction by inpainting for visual anomaly detection [[PR 2021]](https://www.sciencedirect.com/science/article/pii/S0031320320305094/pdfft?md5=9bbe942017de1acd3a97034bc2d4a8fb&pid=1-s2.0-S0031320320305094-main.pdf)
 + Draem-a discriminatively trained reconstruction embedding for surface anomaly detection [[ICCV 2021]](http://arxiv.org/pdf/2108.07610)[[code]](https://github.com/vitjanz/draem)
 + DSR: A dual subspace re-projection network for surface anomaly detection [[ECCV 2022]](https://arxiv.org/pdf/2208.01521.pdf)[[code]](https://github.com/VitjanZ/DSR_anomaly_detection)
 + Natural Synthetic Anomalies for Self-supervised Anomaly Detection and Localization [[ECCV 2022]](https://arxiv.org/pdf/2109.15222.pdf)[[code]](https://github.com/hmsch/natural-synthetic-anomalies)
 + Self-Supervised Training with Autoencoders for Visual Anomaly Detection [[2022]](https://arxiv.org/pdf/2206.11723.pdf)
 + Self-supervised predictive convolutional attentive block for anomaly detection [[CVPR 2022 oral]](http://arxiv.org/pdf/2111.09099)[[code]](https://github.com/ristea/sspcab)
 + Self-Supervised Masked Convolutional Transformer Block for Anomaly Detection [[TPAMI 2022]](https://arxiv.org/pdf/2209.12148.pdf)[[code]](https://github.com/ristea/ssmctb)
 + Iterative energy-based projection on a normal data manifold for anomaly localization [[2019]](https://arxiv.org/pdf/2002.03734.pdf)
 + Towards visually explaining variational autoencoders [[2020]](http://arxiv.org/pdf/1911.07389)
 + Deep generative model using unregularized score for anomaly detection with heterogeneous complexity [[2020]](http://arxiv.org/pdf/1807.05800)
 + Anomaly localization by modeling perceptual features [[2020]](https://arxiv.org/pdf/2008.05369.pdf)
 + Image anomaly detection using normal data only by latent space resampling [[2020]](https://pdfs.semanticscholar.org/cb59/dab0a725c0b511f3140ea47ea0967f3643bf.pdf)
 + Noise-to-Norm Reconstruction for Industrial Anomaly Detection and Localization [[2023]](https://arxiv.org/abs/2307.02836)
 + Patch-wise Auto-Encoder for Visual Anomaly Detection [[2023]](https://arxiv.org/abs/2308.00429)
 + FAIR: Frequency-aware Image Restoration for Industrial Visual Anomaly Detection [[2023]](https://arxiv.org/abs/2309.07068)[[code]](https://github.com/liutongkun/FAIR)
 + Template-guided Hierarchical Feature Restoration for Anomaly Detection [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Guo_Template-guided_Hierarchical_Feature_Restoration_for_Anomaly_Detection_ICCV_2023_paper.pdf)
 + FastRecon: Few-shot Industrial Anomaly Detection via Fast Feature Reconstruction [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Fang_FastRecon_Few-shot_Industrial_Anomaly_Detection_via_Fast_Feature_Reconstruction_ICCV_2023_paper.pdf)[[code]](https://github.com/FzJun26th/FastRecon)
 + Produce Once, Utilize Twice for Anomaly Detection [[2023]](https://arxiv.org/abs/2312.12913)
 + RealNet: A Feature Selection Network with Realistic Synthetic Anomaly for Anomaly Detection [[CVPR 2024]](https://arxiv.org/abs/2403.05897)[[code]](https://github.com/cnulab/RealNet)
 + Implicit Foreground-Guided Network for Anomaly Detection and Localization [[ICASSP 2024]](https://ieeexplore.ieee.org/abstract/document/10446952)
 + Neural Network Training Strategy To Enhance Anomaly Detection Performance: A Perspective On Reconstruction Loss Amplification [[ICASSP 2024]](https://ieeexplore.ieee.org/document/10446942)
 + Patch-Wise Augmentation for Anomaly Detection and Localization [[ICASSP 2024]](https://ieeexplore.ieee.org/document/10446994)
 + A Reconstruction-Based Feature Adaptation for Anomaly Detection with Self-Supervised Multi-Scale Aggregation [[ICASSP 2024]](https://ieeexplore.ieee.org/document/10446766)
 + Mixed-Attention Auto Encoder for Multi-Class Industrial Anomaly Detection [[ICASSP 2024]](https://ieeexplore.ieee.org/document/10446794)
 + Dual-Constraint Autoencoder and Adaptive Weighted Similarity Spatial Attention for Unsupervised Anomaly Detection [[TII 2024]](https://ieeexplore.ieee.org/abstract/document/10504620)
 + Multi-feature Reconstruction Network using Crossed-mask Restoration for Unsupervised Anomaly Detection [[2024]](https://arxiv.org/abs/2404.13273)
 + R3D-AD: Reconstruction via Diffusion for 3D Anomaly Detection [[ECCV 2024]](https://arxiv.org/abs/2407.10862)[[homepage]](https://zhouzheyuan.github.io/r3d-ad)
 + Variational Autoencoder for Anomaly Detection: A Comparative Study [[2024]](https://arxiv.org/abs/2408.13561)[[code]](https://github.com/endtheme123/VAE-compare)
 + Revitalizing Reconstruction Models for Multi-class Anomaly Detection via Class-Aware Contrastive Learning [[2024]](https://arxiv.org/abs/2412.04769)[[code]](https://github.com/LGC-AD/AD-LGC)

### 2.2.2 Generative Adversarial Networks (GANs)
 + Omni-frequency Channel-selection Representations for Unsupervised Anomaly Detection [[TIP 2023]](https://ieeexplore.ieee.org/abstract/document/10192551/)[[code]](https://github.com/zhangzjn/ocr-gan)
 + Learning semantic context from normal samples for unsupervised anomaly detection [[AAAI 2021]](https://ojs.aaai.org/index.php/AAAI/article/download/16420/16227)
 + Anoseg: Anomaly segmentation network using self-supervised learning [[2021]](https://arxiv.org/pdf/2110.03396.pdf)
 + A Surface Defect Detection Method Based on Positive Samples [[PRICAI 2018]](https://link.springer.com/chapter/10.1007/978-3-319-97310-4_54)
 + Few-shot defect image generation via defect-aware feature manipulation [[AAAI 2023]](https://arxiv.org/abs/2303.02389)[[code]](https://github.com/Ldhlwh/DFMGAN)
 + CKAAD: Boosting Fine-Grained Visual Anomaly Detection with Coarse-Knowledge-Aware Adversarial Learning [[AAAI 2025]](https://arxiv.org/abs/2412.12850)[[code]](https://github.com/Faustinaqq/CKAAD)

### 2.2.3 Transformer
 + VT-ADL: A vision transformer network for image anomaly detection and localization [[ISIE 2021]](http://arxiv.org/pdf/2104.10036)
 + ADTR: Anomaly Detection Transformer with Feature Reconstruction [[2022]](https://arxiv.org/pdf/2209.01816.pdf)
 + AnoViT: Unsupervised Anomaly Detection and Localization With Vision Transformer-Based Encoder-Decoder [[2022]](https://ieeexplore.ieee.org/ielx7/6287639/6514899/09765986.pdf)
 + HaloAE: An HaloNet based Local Transformer Auto-Encoder for Anomaly Detection and Localization [[2022]](https://arxiv.org/pdf/2208.03486.pdf)
 + Inpainting transformer for anomaly detection [[ICIAP 2022]](https://arxiv.org/pdf/2104.13897.pdf)
 + Masked Swin Transformer Unet for Industrial Anomaly Detection [[2022]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9858596)
 + Masked Transformer for image Anomaly Localization [[TII 2022]](http://arxiv.org/pdf/2210.15540)
 + Focus the Discrepancy: Intra- and Inter-Correlation Learning for Image Anomaly Detection [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Yao_Focus_the_Discrepancy_Intra-_and_Inter-Correlation_Learning_for_Image_Anomaly_ICCV_2023_paper.pdf)[[code]](https://github.com/xcyao00/FOD)
 + AMI-Net: Adaptive Mask Inpainting Network for Industrial Anomaly Detection and Localization [[TASE 2024]](https://ieeexplore.ieee.org/abstract/document/10445116)
 + Prior Normality Prompt Transformer for Multi-class Industrial Image Anomaly Detection [[TII 2024]](https://arxiv.org/abs/2406.11507)
 + Context Enhancement with Reconstruction as Sequence for Unified Unsupervised Anomaly Detection[[2024]](https://arxiv.org/abs/2409.06285)[[code]](https://github.com/Nothingtolose9979/RAS)
 + Multi-scale feature reconstruction network for industrial anomaly detection [[KBS 2024]](https://www.sciencedirect.com/science/article/pii/S095070512401284X)[[code]](https://github.com/Ehteshamciitwah/MSFR)

### 2.2.4 Diffusion Model
 + AnoDDPM: Anomaly Detection With Denoising Diffusion Probabilistic Models Using Simplex Noise [[CVPR Workshop 2022]](http://dro.dur.ac.uk/36134/1/36134.pdf)
 + Unsupervised Visual Defect Detection with Score-Based Generative Model[[2022]](https://arxiv.org/pdf/2211.16092.pdf)
 + DiffusionAD: Denoising Diffusion for Anomaly Detection [[2023]](https://arxiv.org/abs/2303.08730)[[code]](https://github.com/HuiZhang0812/DiffusionAD)
 + Anomaly Detection with Conditioned Denoising Diffusion Models [[2023]](https://arxiv.org/abs/2305.15956)
 + Unsupervised Surface Anomaly Detection with Diffusion Probabilistic Model [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Unsupervised_Surface_Anomaly_Detection_with_Diffusion_Probabilistic_Model_ICCV_2023_paper.pdf)
 + Removing Anomalies as Noises for Industrial Defect Localization [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Lu_Removing_Anomalies_as_Noises_for_Industrial_Defect_Localization_ICCV_2023_paper.pdf)
 + TransFusion -- A Transparency-Based Diffusion Model for Anomaly Detection [[ECCV 2024]](https://arxiv.org/abs/2311.09999)[[code]](https://github.com/MaticFuc/ECCV_TransFusion)
 + LafitE: Latent Diffusion Model with Feature Editing for Unsupervised Multi-class Anomaly Detection [[2023]](https://arxiv.org/abs/2307.08059)
 + DiAD: A Diffusion-based Framework for Multi-class Anomaly Detection [[AAAI 2024]](https://ojs.aaai.org/index.php/AAAI/article/view/28690)[[code]](https://lewandofskee.github.io/projects/diad)
 + D3AD: Dynamic Denoising Diffusion Probabilistic Model for Anomaly Detection [[2024]](https://arxiv.org/abs/2401.04463)
 + GLAD: Towards Better Reconstruction with Global and Local Adaptive Diffusion Models for Unsupervised Anomaly Detection [[ECCV 2024]](https://arxiv.org/abs/2406.07487)[[code]](https://github.com/hyao1/GLAD)
 + Tackling Structural Hallucination in Image Translation with Local Diffusion [[ECCV 2024 oral]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10498.pdf)[[code]](https://github.com/edshkim98/LocalDiffusion-Hallucination)

### 2.2.5 Others
 + Anomaly Detection using Score-based Perturbation Resilience [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Shin_Anomaly_Detection_using_Score-based_Perturbation_Resilience_ICCV_2023_paper.pdf)

## 2.3 Supervised AD
### More Normal Samples With (Less Abnormal Samples or Weak Labels)
+ Neural batch sampling with reinforcement learning for semi-supervised anomaly detection [[ECCV 2020]](https://www.ri.cmu.edu/wp-content/uploads/2020/05/WenHsuan_MSR_Thesis-1.pdf)
+ Explainable Deep One-Class Classification [[ICLR 2020]](https://arxiv.org/pdf/2007.01760.pdf)
+ Attention guided anomaly localization in images [[ECCV 2020]](http://arxiv.org/pdf/1911.08616)
+ Mixed supervision for surface-defect detection: From weakly to fully supervised learning [[2021]](https://arxiv.org/pdf/2104.06064.pdf)
+ Explainable deep few-shot anomaly detection with deviation networks [[2021]](https://arxiv.org/pdf/2108.00462.pdf)[[code]](https://github.com/Choubo/deviation-network-image)
+ Catching Both Gray and Black Swans: Open-set Supervised Anomaly Detection [[CVPR 2022]](http://arxiv.org/pdf/2203.14506)[[code]](https://github.com/Choubo/DRA)
+ Anomaly Clustering: Grouping Images into Coherent Clusters of Anomaly Types[[WACV 2023]](https://openaccess.thecvf.com/content/WACV2023/html/Sohn_Anomaly_Clustering_Grouping_Images_Into_Coherent_Clusters_of_Anomaly_Types_WACV_2023_paper.html)
+ Prototypical Residual Networks for Anomaly Detection and Localization [[CVPR 2023]](https://arxiv.org/abs/2212.02031)[[code]](https://github.com/xcyao00/PRNet)
+ Efficient Anomaly Detection with Budget Annotation Using Semi-Supervised Residual Transformer [[2023]](https://arxiv.org/abs/2306.03492)
+ Anomaly Heterogeneity Learning for Open-set Supervised Anomaly Detection [[CVPR 2024]](https://arxiv.org/abs/2310.12790)[[code]](https://github.com/mala-lab/AHL)
+ Few-shot defect image generation via defect-aware feature manipulation [[AAAI 2023]](https://arxiv.org/abs/2303.02389)[[code]](https://github.com/Ldhlwh/DFMGAN)
+ AnomalyDiffusion: Few-Shot Anomaly Image Generation with Diffusion Model [[AAAI 2024]](https://ojs.aaai.org/index.php/AAAI/article/view/28696)[[code]](https://github.com/sjtuplayer/anomalydiffusion)
+ BiaS: Incorporating Biased Knowledge to Boost Unsupervised Image Anomaly Localization [[TSMC 2024]](https://ieeexplore.ieee.org/abstract/document/10402554)
+ DMAD: Dual Memory Bank for Real-World Anomaly Detection [[2024]](https://arxiv.org/abs/2403.12362)
+ AnomalousPatchCore: Exploring the Use of Anomalous Samples in Industrial Anomaly Detection [[ECCVW 2024]](https://arxiv.org/abs/2408.15113)
+ SuperSimpleNet: Unifying Unsupervised and Supervised Learning for Fast and Reliable Surface Defect Detection [[ICIP 2024]](https://arxiv.org/abs/2408.03143)[[code]](https://github.com/blaz-r/SuperSimpleNet/tree/main)
+ VarAD: Lightweight High-Resolution Image Anomaly Detection via Visual Autoregressive Modeling [[TII 2025]](https://arxiv.org/abs/2412.17263)[[code]](https://github.com/caoyunkang/VarAD)

### More Abnormal Samples
+ Logit Inducing With Abnormality Capturing for Semi-Supervised Image Anomaly Detection [[2022]](https://ieeexplore.ieee.org/document/9885240)
+ An effective framework of automated visual surface defect detection for metal parts [[2021]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9475966)
+ Interleaved Deep Artifacts-Aware Attention Mechanism for Concrete Structural Defect Classification [[TIP 2021]](https://eprints.keele.ac.uk/10031/1/TIP24Jul2021.pdf)
+ Reference-based defect detection network [[TIP 2021]](http://arxiv.org/pdf/2108.04456)
+ Fabric defect detection using tactile information [[ICRA 2021]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9561092)
+ A lightweight spatial and temporal multi-feature fusion network for defect detection [[TIP 2020]](http://nrl.northumbria.ac.uk/id/eprint/48908/1/ALightweightSpatialandTemporalMulti-featureFusionNetworkforDefectDetection.pdf)
+ SDD-CNN: Small Data-Driven Convolution Neural Networks for Subtle Roller Defect Inspection [[Robotics and Computer-Integrated Manufacturing 2020]](https://www.sciencedirect.com/science/article/abs/pii/S0736584518304770)
+ A High-Efficiency Fully Convolutional Networks for Pixel-Wise Surface Defect Detection [[IEEE Access 2019]](https://ieeexplore.ieee.org/abstract/document/8624360)
+ SDD-CNN: Small Data-Driven Convolution Neural Networks for Subtle Roller Defect Inspection [[Applied Sciences 2019]](https://www.mdpi.com/2076-3417/9/7/1364)
+ Autonomous Structural Visual Inspection Using Region-Based Deep Learning for Detecting Multiple Damage Types [[CACIE 2018]](https://dl.acm.org/doi/abs/10.1111/mice.12334)
+ Detection and segmentation of manufacturing defects with convolutional neural networks and transfer learning [[2018]](https://europepmc.org/articles/pmc6512995?pdf=render)
+ Automatic Metallic Surface Defect Detection and Recognition with Convolutional Neural Networks [[Applied Sciences 2018]](https://www.mdpi.com/2076-3417/8/9/1575)
+ Real-time Detection of Steel Strip Surface Defects Based on Improved YOLO Detection Network [[IFAC-PapersOnLine 2018]](https://www.sciencedirect.com/science/article/pii/S2405896318321001)
+ Domain adaptation for automatic OLED panel defect detection using adaptive support vector data description [[IJCV 2017]](https://link.springer.com/article/10.1007/s11263-016-0953-y)
+ Automatic Defect Detection of Fasteners on the Catenary Support Device Using Deep Convolutional Neural Network [[TIM 2017]](https://ieeexplore.ieee.org/abstract/document/8126877)
+ Deep Active Learning for Civil Infrastructure Defect Detection and Classification [[Computing in civil engineering 2017]](https://ascelibrary.org/doi/abs/10.1061/9780784480823.036)
+ A fast and robust convolutional neural network-based defect detection model in product quality control [[IJAMT 2017]](https://link.springer.com/article/10.1007/s00170-017-0882-0)
+ Defects Detection Based on Deep Learning and Transfer Learning [[Metallurgical & Mining Industry 2015]](https://web.s.ebscohost.com/abstract?direct=true&profile=ehost&scope=site&authtype=crawler&jrnl=20760507&AN=115932631&h=Xxf%2binGAfPaFG1E3Net%2fQQIu5U%2fD2pFkichv9fJ63Bx%2bjW2wr5y1UZWYaHbOQCE%2bZc%2bYJQz117Xd06J3IxAbSg%3d%3d&crl=c&resultNs=AdminWebAuth&resultLocal=ErrCrlNotAuth&crlhashurl=login.aspx%3fdirect%3dtrue%26profile%3dehost%26scope%3dsite%26authtype%3dcrawler%26jrnl%3d20760507%26AN%3d115932631)
+ Design of deep convolutional neural network architectures for automated feature extraction in industrial inspection [[CIRP annals 2016]](https://www.sciencedirect.com/science/article/abs/pii/S0007850616300725)
+ Decision Fusion Network with Perception Fine-tuning for Defect Classification [[2023]](https://arxiv.org/abs/2309.12630)
+ Global Context Aggregation Network for Lightweight Saliency Detection of Surface Defects [[2023]](https://arxiv.org/abs/2309.12641)
+ Dual Attention U-Net with Feature Infusion: Pushing the Boundaries of Multiclass Defect Segmentation [[2023]](https://arxiv.org/abs/2312.14053)[[code]](https://github.com/RashaAlshawi/Dual-Attention-U-Net-with-Feature-Infusion-Pushing-the-Boundaries-of-Multiclass-Defect-Segmentation)
+ MemoryMamba: Memory-Augmented State Space Model for Defect Recognition [[2024]](https://arxiv.org/abs/2405.03673)
+ Supervised Anomaly Detection for Complex Industrial Images [[2024]](https://arxiv.org/abs/2405.04953)[[code]](https://github.com/abc-125/segad)
+ Small Object Few-shot Segmentation for Vision-based Industrial Inspection [[2024]](https://arxiv.org/abs/2407.21351)[[code]](https://github.com/zhangzilongc/SOFS)

# 3 Other Research Direction

## 3.1 Zero/Few-Shot AD

### Zero-Shot AD
 + Random Word Data Augmentation with CLIP for Zero-Shot Anomaly Detection [[BMVC 2023]](https://arxiv.org/abs/2308.11119)
 + Zero-Shot Batch-Level Anomaly Detection [[2023]](https://arxiv.org/abs/2302.07849)
 + Zero-shot versus Many-shot: Unsupervised Texture Anomaly Detection [[WACV 2023]](https://ieeexplore.ieee.org/document/10030870)
 + MAEDAY: MAE for few and zero shot AnomalY-Detection [[2022]](https://arxiv.org/pdf/2211.14307.pdf)
 + WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation [[CVPR 2023]](https://arxiv.org/abs/2303.14814) [[unofficial code in AnomalyCLIP]](https://github.com/zqhang/Accurate-WinCLIP-pytorch) [[unofficial code in SAA]](https://github.com/caoyunkang/WinClip) [[unofficial code in mala-lab]](https://github.com/mala-lab/WinCLIP)
 + Segment Any Anomaly without Training via Hybrid Prompt Regularization [[2023]](https://arxiv.org/abs/2305.10724) [[code]](https://github.com/caoyunkang/GroundedSAM-zero-shot-anomaly-detection)
 + Anomaly Detection in an Open World by a Neuro-symbolic Program on Zero-shot Symbols [[IROS 2022 Workshop]](https://openreview.net/pdf?id=Bg3ZO3nXJuA)
 + AnoVL: Adapting Vision-Language Models for Unified Zero-shot Anomaly Localization [[2023]](https://arxiv.org/abs/2308.15939)[[code]](https://github.com/hq-deng/AnoVL)
 + CLIP-AD: A Language-Guided Staged Dual-Path Model for Zero-shot Anomaly Detection [[2023]](https://arxiv.org/abs/2311.00453)
 + PromptAD: Zero-shot Anomaly Detection using Text Prompts [[WACV 2024]](https://openaccess.thecvf.com/content/WACV2024/papers/Li_PromptAD_Zero-Shot_Anomaly_Detection_Using_Text_Prompts_WACV_2024_paper.pdf)
 + High-Fidelity Zero-Shot Texture Anomaly Localization Using Feature Correspondence Analysis [[WACV 2024]](https://openaccess.thecvf.com/content/WACV2024/html/Ardelean_High-Fidelity_Zero-Shot_Texture_Anomaly_Localization_Using_Feature_Correspondence_Analysis_WACV_2024_paper.html)
 + AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection [[ICLR 2024]](https://openreview.net/forum?id=buC4E91xZE)[[code]](https://github.com/zqhang/AnomalyCLIP)
 + MuSc: Zero-Shot Industrial Anomaly Classification and Segmentation with Mutual Scoring of the Unlabeled Images[[ICLR 2024]](https://openreview.net/forum?id=AHgc5SMdtd)[[code]](https://github.com/xrli-U/MuSc)
 + ClipSAM: CLIP and SAM Collaboration for Zero-Shot Anomaly Segmentation [[2023]](https://arxiv.org/pdf/2401.12665)
 + APRIL-GAN: A Zero-/Few-Shot Anomaly Classification and Segmentation Method for CVPR 2023 VAND Workshop Challenge Tracks 1&2: 1st Place on Zero-shot AD and 4th Place on Few-shot AD [[CVPRW 2023]](https://arxiv.org/abs/2305.17382)[[code]](https://github.com/ByChelsea/VAND-APRIL-GAN)
 + Model Selection of Zero-shot Anomaly Detectors in the Absence of Labeled Validation Data [[2024]](https://arxiv.org/abs/2310.10461)
 + PromptAD: Learning Prompts with only Normal Samples for Few-Shot Anomaly Detection [[CVPR 2024]](https://arxiv.org/abs/2404.05231)[[code]](https://github.com/FuNz-0/PromptAD)
 + Do LLMs Understand Visual Anomalies? Uncovering LLM Capabilities in Zero-shot Anomaly Detection [[2024]](https://arxiv.org/abs/2404.09654)
 + FiLo: Zero-Shot Anomaly Detection by Fine-Grained Description and High-Quality Localization [[2024]](https://arxiv.org/abs/2404.13671)
 + Dual-Image Enhanced CLIP for Zero-Shot Anomaly Detection [[2024]](https://arxiv.org/abs/2405.04782)
 + Investigating the Semantic Robustness of CLIP-based Zero-Shot Anomaly Segmentation [[2024]](https://arxiv.org/abs/2405.07969)
 + SAM-LAD: Segment Anything Model Meets Zero-Shot Logic Anomaly Detection [[2024]](https://arxiv.org/abs/2406.00625)
 + VCP-CLIP: A visual context prompting model for zero-shot anomaly segmentation [[ECCV 2024]](https://arxiv.org/abs/2407.12276)[[code]](https://github.com/xiaozhen228/VCP-CLIP)
 + AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for Zero-Shot Anomaly Detection [[ECCV 2024]](https://arxiv.org/abs/2407.15795)[[code]](https://github.com/caoyunkang/AdaCLIP)
 + Towards Zero-shot Point Cloud Anomaly Detection: A Multi-View Projection Framework [[2024]](https://arxiv.org/abs/2409.13162)
 + PointAD: Comprehending 3D Anomalies from Points and Pixels for Zero-shot 3D Anomaly Detection [[NeurIPS 2024]](https://arxiv.org/abs/2410.00320)[[code]](https://github.com/zqhang/PointAD)
 + VMAD: Visual-enhanced Multimodal Large Language Model for Zero-Shot Anomaly Detection [[2024]](https://arxiv.org/abs/2409.20146)
 + GlocalCLIP: Object-agnostic Global-Local Prompt Learning for Zero-shot Anomaly Detection [[2024]](https://arxiv.org/abs/2411.06071)
 + Towards Zero-shot 3D Anomaly Localization [[WACV 2025]](https://arxiv.org/abs/2412.04304)


### Few-Shot AD
 + Learning unsupervised metaformer for anomaly detection [[ICCV 2021]](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_Learning_Unsupervised_Metaformer_for_Anomaly_Detection_ICCV_2021_paper.pdf)
 + Registration based few-shot anomaly detection [[ECCV 2022 oral]](https://arxiv.org/pdf/2207.07361.pdf)[[code]](https://github.com/MediaBrain-SJTU/RegAD)
 + Same same but differnet: Semi-supervised defect detection with normalizing flows [[(Distribution)WACV 2021]](http://arxiv.org/pdf/2008.12577)
 + Towards total recall in industrial anomaly detection [[(Memory bank)CVPR 2022]](http://arxiv.org/pdf/2106.08265)
 + A hierarchical transformation-discriminating generative model for few shot anomaly detection [[ICCV 2021]](http://arxiv.org/pdf/2104.14535)
 + Anomaly detection of defect using energy of point pattern features within random finite set framework [[2021]](https://arxiv.org/pdf/2108.12159.pdf)
 + Optimizing PatchCore for Few/many-shot Anomaly Detection [[2023]](https://arxiv.org/abs/2307.10792)[[code]](https://github.com/scortexio/patchcore-few-shot/)
 + AnomalyGPT: Detecting Industrial Anomalies using Large Vision-Language Models [[AAAI 2024]](https://arxiv.org/abs/2308.15366)[[code]](https://github.com/CASIA-IVA-Lab/AnomalyGPT)[[project page]](https://anomalygpt.github.io/)
 + FastRecon: Few-shot Industrial Anomaly Detection via Fast Feature Reconstruction [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Fang_FastRecon_Few-shot_Industrial_Anomaly_Detection_via_Fast_Feature_Reconstruction_ICCV_2023_paper.pdf)[[code]](https://github.com/FzJun26th/FastRecon)
 + Produce Once, Utilize Twice for Anomaly Detection [[2023]](https://arxiv.org/abs/2312.12913)
 + COFT-AD: COntrastive Fine-Tuning for Few-Shot Anomaly Detection [[TIP2024]](http://arxiv.org/abs/2402.18998)
 + Text-Guided Variational Image Generation for Industrial Anomaly Detection and Segmentation [[CVPR 2024]](https://arxiv.org/abs/2403.06247)[[code]](https://github.com/MingyuLee82/TGI_AD_v1)
 + Multimodal Industrial Anomaly Detection by Crossmodal Feature Mapping [[CVPR 2024]](https://arxiv.org/abs/2312.04521)
 + Dual-path Frequency Discriminators for Few-shot Anomaly Detection [[2024]](https://arxiv.org/abs/2403.04151)
 + Few-shot Online Anomaly Detection and Segmentation [[2024]](https://arxiv.org/abs/2403.18201)
 + FewSOME: One-Class Few Shot Anomaly Detection with Siamese Networks [[CVPRW 2023]](https://openaccess.thecvf.com/content/CVPR2023W/VAND/papers/Belton_FewSOME_One-Class_Few_Shot_Anomaly_Detection_With_Siamese_Networks_CVPRW_2023_paper.pdf)[[code]](https://github.com/niamhbelton/FewSOME)
 + AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2 [[2024]](https://arxiv.org/abs/2405.14529)
 + Small Object Few-shot Segmentation for Vision-based Industrial Inspection [[2024]](https://arxiv.org/abs/2407.21351)[[code]](https://github.com/zhangzilongc/SOFS)
 + Few-Shot Anomaly Detection via Category-Agnostic Registration Learning [[2024]](https://arxiv.org/abs/2406.08810)[[code]](https://github.com/Haoyan-Guan/CAReg)
 + AnoPLe: Few-Shot Anomaly Detection via Bi-directional Prompt Learning with Only Normal Samples [[2024]](https://arxiv.org/abs/2408.13516)[[code]](https://github.com/YoojLee/AnoPLe)
 + InCTRL: Toward Generalist Anomaly Detection via In-context Residual Learning with Few-shot Sample Prompts [[CVPR 2024]](https://arxiv.org/abs/2403.06495)[[code]](https://github.com/mala-lab/InCTRL)
 + FADE: Few-shot/zero-shot Anomaly Detection Engine using Large Vision-Language Model[[BMVC 2024]](https://arxiv.org/abs/2409.00556#)[[code]](https://github.com/BMVC-FADE/BMVC-FADE)
 + FOCT: Few-shot Industrial Anomaly Detection with Foreground-aware Online Conditional Transport [[ACM MM 2024]](https://dl.acm.org/doi/10.1145/3664647.3680771)
 + UniVAD: A Training-free Unified Model for Few-shot Visual Anomaly Detection [[2024]](https://arxiv.org/abs/2412.03342)[[code]](https://uni-vad.github.io/#)
 + SOWA: Adapting Hierarchical Frozen Window Self-Attention to Visual-Language Models for Better Anomaly Detection [[2024]](https://arxiv.org/abs/2407.03634)[[code]](https://github.com/huzongxiang/sowa)
 + CLIP-FSAC++: Few-Shot Anomaly Classification with Anomaly Descriptor Based on CLIP [[2024]](https://arxiv.org/abs/2412.03829)[[code]](https://github.com/Jay-zzcoder/clip-fsac-pp)
 + KAG-prompt: Kernel-Aware Graph Prompt Learning for Few-Shot Anomaly Detection [[AAAI 2025]](https://arxiv.org/abs/2412.17619)[[code]](https://github.com/CVL-hub/KAG-prompt)


## 3.2 Noisy AD
 + Trustmae: A noise-resilient defect classification framework using memory-augmented auto-encoders with trust regions [[WACV 2021]](http://arxiv.org/pdf/2012.14629)
 + Self-Supervise, Refine, Repeat: Improving Unsupervised Anomaly Detection [[TMLR 2021]](https://arxiv.org/pdf/2106.06115.pdf)
 + Data refinement for fully unsupervised visual inspection using pre-trained networks [[2022]](https://arxiv.org/pdf/2202.12759.pdf)
 + Latent Outlier Exposure for Anomaly Detection with Contaminated Data [[ICML 2022]](https://arxiv.org/pdf/2202.08088.pdf)
 + Deep one-class classification via interpolated gaussian descriptor [[AAAI 2022 oral]](https://arxiv.org/pdf/2101.10043.pdf)[[code]](https://github.com/tianyu0207/IGD)
 + SoftPatch: Unsupervised Anomaly Detection with Noisy Data [[NeurIPS 2022]](https://openreview.net/pdf?id=pIYYJflkhZ)[[code]](https://github.com/TencentYoutuResearch/AnomalyDetection-SoftPatch)
 + Inter-Realization Channels: Unsupervised Anomaly Detection Beyond One-Class Classification [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/McIntosh_Inter-Realization_Channels_Unsupervised_Anomaly_Detection_Beyond_One-Class_Classification_ICCV_2023_paper.pdf)[[code]](https://github.com/DeclanMcIntosh/InReaCh)
 + M3DM-NR: RGB-3D Noisy-Resistant Industrial Anomaly Detection via Multimodal Denoising [[2024]](https://arxiv.org/abs/2406.02263)
 + SoftPatch+: Fully Unsupervised Anomaly Classification and Segmentation [[2024]](https://arxiv.org/abs/2412.20870)[[code]](https://github.com/TencentYoutuResearch/AnomalyDetection-SoftPatch)

## 3.3 Anomaly Synthetic
 + Cutpaste: Self-supervised learning for anomaly detection and localization [[(OCC)ICCV 2021]](http://arxiv.org/pdf/2104.04015)[[unofficial code]](https://github.com/Runinho/pytorch-cutpaste)
 + Draem-a discriminatively trained reconstruction embedding for surface anomaly detection [[(Reconstruction AE)ICCV 2021]](http://arxiv.org/pdf/2108.07610)[[code]](https://github.com/vitjanz/draem)
 + DSR: A dual subspace re-projection network for surface anomaly detection [[ECCV 2022]](https://arxiv.org/pdf/2208.01521.pdf)[[code]](https://github.com/VitjanZ/DSR_anomaly_detection)
 + Natural Synthetic Anomalies for Self-supervised Anomaly Detection and Localization [[ECCV 2022]](https://arxiv.org/pdf/2109.15222.pdf)[[code]](https://github.com/hmsch/natural-synthetic-anomalies)
 + MemSeg: A semi-supervised method for image surface defect detection using differences and commonalities [[(OCC)2022]](https://arxiv.org/pdf/2205.00908.pdf)[[unofficial code]](https://github.com/TooTouch/MemSeg)
 + A High-Efficiency Fully Convolutional Networks for Pixel-Wise Surface Defect Detection [[IEEE Access 2019]](https://ieeexplore.ieee.org/abstract/document/8624360)
 + Multistage GAN for fabric defect detection [[2019]](https://pubmed.ncbi.nlm.nih.gov/31870985/)
 + Gan-based defect synthesis for anomaly detection in fabrics [[2020]](https://www.lfb.rwth-aachen.de/bibtexupload/pdf/RIP20c.pdf)
 + Defect image sample generation with GAN for improving defect recognition [[2020]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9000806)
 + Defective samples simulation through neural style transfer for automatic surface defect segment [[2020]](http://arxiv.org/pdf/1910.03334)
 + A simulation-based few samples learning method for surface defect segmentation [[2020]](https://www.sciencedirect.com/science/article/pii/S0925231220310791/pdfft?md5=f3f72bc8687c8f9968d4a2a1bd3ea17e&pid=1-s2.0-S0925231220310791-main.pdf)
 + Synthetic data augmentation for surface defect detection and classification using deep learning [[2020]](https://link.springer.com/article/10.1007/s10845-020-01710-x)
 + Defect Transfer GAN: Diverse Defect Synthesis for Data Augmentation [[BMVC 2022]](https://openreview.net/pdf?id=2hMEdc35xZ6)
 + Defect-GAN: High-fidelity defect synthesis for automated defect inspection [[2021]](https://dr.ntu.edu.sg/bitstream/10356/146285/2/WACV_2021_Defect_GAN__Camera_Ready_.pdf)
 + EID-GAN: Generative Adversarial Nets for Extremely Imbalanced Data Augmentation[[TII 2022]](https://ieeexplore.ieee.org/document/9795891)
 + Multilevel Saliency-Guided Self-Supervised Learning for Image Anomaly Detection [[2023]](http://arxiv.org/pdf/2311.18332v1)
 + DeSTSeg: Segmentation Guided Denoising Student-Teacher for Anomaly Detection [[CVPR 2023]](https://arxiv.org/abs/2211.11317)[[code]](https://github.com/apple/ml-destseg)
 + AnomalyDiffusion: Few-Shot Anomaly Image Generation with Diffusion Model [[AAAI 2024]](https://ojs.aaai.org/index.php/AAAI/article/view/28696)[[code]](https://github.com/sjtuplayer/anomalydiffusion)
 + RealNet: A Feature Selection Network with Realistic Synthetic Anomaly for Anomaly Detection [[CVPR 2024]](https://arxiv.org/abs/2403.05897)[[code]](https://github.com/cnulab/RealNet)
 + Dual-path Frequency Discriminators for Few-shot Anomaly Detection [[2024]](https://arxiv.org/abs/2403.04151)
 + A Novel Approach to Industrial Defect Generation through Blended Latent Diffusion Model with Online Adaptation [[2024]](https://arxiv.org/abs/2402.19330)[[code]](https://github.com/GrandpaXun242/AdaBLDM)
 + A Comprehensive Augmentation Framework for Anomaly Detection [[AAAI 2024]](https://ojs.aaai.org/index.php/AAAI/article/view/28720)
 + CAGEN: Controllable Anomaly Generator using Diffusion Model [[ICASSP 2024]](https://ieeexplore.ieee.org/document/10447663)
 + AnomalyXFusion: Multi-modal Anomaly Synthesis with Diffusion [[2024]](https://arxiv.org/abs/2404.19444)[[data]](https://github.com/hujiecpp/MVTec-Caption)
 + Few-shot defect image generation via defect-aware feature manipulation [[AAAI 2023]](https://arxiv.org/abs/2303.02389)[[code]](https://github.com/Ldhlwh/DFMGAN)
 + A Unified Anomaly Synthesis Strategy with Gradient Ascent for Industrial Anomaly Detection and Localization [[ECCV 2024]](https://arxiv.org/abs/2407.09359)[[code]](https://github.com/cqylunlun/GLASS)
 + SLSG: Industrial Image Anomaly Detection with Improved Feature Embeddings and One-Class Classification [[PR 2024]](https://www.sciencedirect.com/science/article/pii/S0031320324006137)
 + Dual-Modeling Decouple Distillation for Unsupervised Anomaly Detection [[ACM MM 2024]](https://arxiv.org/abs/2408.03888)
 + SuperSimpleNet: Unifying Unsupervised and Supervised Learning for Fast and Reliable Surface Defect Detection [[ICIP 2024]](https://arxiv.org/abs/2408.03143)[[code]](https://github.com/blaz-r/SuperSimpleNet/tree/main)
 + AnomalyControl: Learning Cross-modal Semantic Features for Controllable Anomaly Synthesis [[2024]](https://arxiv.org/abs/2412.06510)
 + Progressive Boundary Guided Anomaly Synthesis for Industrial Anomaly Detection [[TCSVT 2024]](https://ieeexplore.ieee.org/document/10716437)[[code]](https://github.com/cqylunlun/PBAS)

## 3.4 RGBD AD
 + Anomaly detection in 3d point clouds using deep geometric descriptors [[WACV 2022]](https://arxiv.org/pdf/2202.11660.pdf)
 + Back to the feature: classical 3d features are (almost) all you need for 3D anomaly detection [[2022]](https://arxiv.org/pdf/2203.05550.pdf)[[code]](https://github.com/eliahuhorwitz/3D-ADS)
 + Anomaly Detection Requires Better Representations [[2022]](https://arxiv.org/pdf/2210.10773.pdf)
 + Asymmetric Student-Teacher Networks for Industrial Anomaly Detection [[WACV 2022]](https://arxiv.org/pdf/2210.07829.pdf)
 + Multimodal Industrial Anomaly Detection via Hybrid Fusion [[CVPR 2023]](https://arxiv.org/abs/2303.00601)[[code]](https://github.com/nomewang/M3DM)
 + Complementary Pseudo Multimodal Feature for Point Cloud Anomaly Detection [[2023]](https://arxiv.org/abs/2303.13194)[[code]](https://github.com/caoyunkang/CPMF)
 + Image-Pointcloud Fusion based Anomaly Detection using PD-REAL Dataset [[2023]](https://arxiv.org/abs/2311.04095)[[data]](https://github.com/Andy-cs008/PD-REAL)
 + Towards Scalable 3D Anomaly Detection and Localization: A Benchmark via 3D Anomaly Synthesis and A Self-Supervised Learning Network [[CVPR 2024]](https://arxiv.org/abs/2311.14897)[[code]](https://github.com/Chopper-233/Anomaly-ShapeNet)
 + Shape-Guided Dual-Memory Learning for 3D Anomaly Detection [[ICML 2023]](https://openreview.net/forum?id=IkSGn9fcPz)
 + EasyNet: An Easy Network for 3D Industrial Anomaly Detection [[ACM MM 2023]](https://arxiv.org/abs/2307.13925)
 + Self-supervised Feature Adaptation for 3D Industrial Anomaly Detection [[2024]](https://arxiv.org/abs/2401.03145)
 + Cheating Depth: Enhancing 3D Surface Anomaly Detection via Depth Simulation [[WACV 2024]](https://openaccess.thecvf.com/content/WACV2024/papers/Zavrtanik_Cheating_Depth_Enhancing_3D_Surface_Anomaly_Detection_via_Depth_Simulation_WACV_2024_paper.pdf)[[code]](https://github.com/VitjanZ/3DSR)
 + Incremental Template Neighborhood Matching for 3D anomaly detection [[Neurocomputing 2024]](https://www.sciencedirect.com/science/article/abs/pii/S0925231224002546)
 + Keep DRÆMing: Discriminative 3D anomaly detection through anomaly simulation [[PRL 2024]](https://www.sciencedirect.com/science/article/pii/S0167865524000862)
 + Rethinking Reverse Distillation for Multi-Modal Anomaly Detection [[AAAI 2024]](https://ojs.aaai.org/index.php/AAAI/article/view/28687)
 + Multimodal Industrial Anomaly Detection by Crossmodal Feature Mapping [[CVPR 2024]](https://arxiv.org/abs/2312.04521)
 + Cross-Modal Distillation in Industrial Anomaly Detection: Exploring Efficient Multi-Modal IAD [[2024]](https://arxiv.org/abs/2405.13571)[[code]](https://github.com/evenrose/CMDIAD)
 + M3DM-NR: RGB-3D Noisy-Resistant Industrial Anomaly Detection via Multimodal Denoising [[2024]](https://arxiv.org/abs/2406.02263)
 + Towards Zero-shot 3D Anomaly Localization [[WACV 2025]](https://arxiv.org/abs/2412.04304)
 + Revisiting Multimodal Fusion for 3D Anomaly Detection from an Architectural Perspective [[AAAI 2025]](https://arxiv.org/abs/2412.17297)

## 3.5 3D AD
 + Real3D-AD: A Dataset of Point Cloud Anomaly Detection [[NeurIPS 2023]](https://arxiv.org/abs/2309.13226)[[code]](https://github.com/M-3LAB/Real3D-AD)
 + PointCore: Efficient Unsupervised Point Cloud Anomaly Detector Using Local-Global Features [[2024]](https://arxiv.org/abs/2403.01804)
 + Towards Scalable 3D Anomaly Detection and Localization: A Benchmark via 3D Anomaly Synthesis and A Self-Supervised Learning Network [[CVPR 2024]](https://arxiv.org/abs/2311.14897)[[code]](https://github.com/Chopper-233/Anomaly-ShapeNet)
 + R3D-AD: Reconstruction via Diffusion for 3D Anomaly Detection [[ECCV 2024]](https://arxiv.org/abs/2407.10862)[[homepage]](https://zhouzheyuan.github.io/r3d-ad)
 + Towards High-resolution 3D Anomaly Detection via Group-Level Feature Contrastive Learning [[ACM MM 2024]](https://arxiv.org/abs/2408.04604)[[code]](https://github.com/M-3LAB/Group3AD)
 + Complementary Pseudo Multimodal Feature for Point Cloud Anomaly Detection [[PR 2024]](https://www.sciencedirect.com/science/article/abs/pii/S0031320324005120) [[code]](https://github.com/caoyunkang/CPMF)
 + Towards Zero-shot Point Cloud Anomaly Detection: A Multi-View Projection Framework [[2024]](https://arxiv.org/abs/2409.13162)[[code]](https://github.com/hustCYQ/MVP-PCLIP)
 + MulSen-AD: A Dataset and Benchmark for Multi-Sensor Anomaly Detection [[2024]](https://zzzbbbzzz.github.io/MulSen_AD/index.html)[[code]](https://github.com/ZZZBBBZZZ/MulSen-AD)
 + PointAD: Comprehending 3D Anomalies from Points and Pixels for Zero-shot 3D Anomaly Detection [[NeurIPS 2024]](https://arxiv.org/abs/2410.00320)[[code]](https://github.com/zqhang/PointAD)
 + Look Inside for More: Internal Spatial Modality Perception for 3D Anomaly Detection [[AAAI 2025]](https://arxiv.org/abs/2412.13461)

## 3.6 Continual AD
 + Towards Total Online Unsupervised Anomaly Detection and Localization in Industrial Vision [[2023]](https://arxiv.org/abs/2305.15652)
 + Towards Continual Adaptation in Industrial Anomaly Detection [[ACM MM 2022]](https://dl.acm.org/doi/abs/10.1145/3503161.3548232)
 + An Incremental Unified Framework for Small Defect Inspection [[2023]](https://arxiv.org/abs/2312.08917)[[code]](https://github.com/jqtangust/IUF)
 + Unsupervised Continual Anomaly Detection with Contrastively-learned Prompt [[AAAI 2024]](https://ojs.aaai.org/index.php/AAAI/article/view/28153)[[code]](https://github.com/shirowalker/UCAD)

## 3.7 Uniform/Multi-Class AD
 + A Unified Model for Multi-class Anomaly Detection [[NeurIPS 2022]](https://arxiv.org/pdf/2206.03687.pdf) [[code]](https://github.com/zhiyuanyou/UniAD)
 + OmniAL A unifiled CNN framework for unsupervised anomaly localization [[CVPR 2023]](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_OmniAL_A_Unified_CNN_Framework_for_Unsupervised_Anomaly_Localization_CVPR_2023_paper.pdf)
 + SelFormaly: Towards Task-Agnostic Unified Anomaly Detection[[2023]](https://arxiv.org/abs/2307.12540)
 + Hierarchical Vector Quantized Transformer for Multi-class Unsupervised Anomaly Detection [[NeurIPS 2023]](https://openreview.net/pdf?id=clJTNssgn6)[[code]](https://github.com/RuiyingLu/HVQ-Trans)
 + Removing Anomalies as Noises for Industrial Defect Localization [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Lu_Removing_Anomalies_as_Noises_for_Industrial_Defect_Localization_ICCV_2023_paper.pdf)
 + UniFormaly: Towards Task-Agnostic Unified Framework for Visual Anomaly Detection [[2023]](https://arxiv.org/abs/2307.12540)[[code]](https://github.com/YoojLee/Uniformaly)
 + MSTAD: A masked subspace-like transformer for multi-class anomaly detection [[2023]](https://www.sciencedirect.com/science/article/pii/S095070512300936X)
 + LafitE: Latent Diffusion Model with Feature Editing for Unsupervised Multi-class Anomaly Detection [[2023]](https://arxiv.org/abs/2307.08059)
 + DiAD: A Diffusion-based Framework for Multi-class Anomaly Detection [[AAAI 2024]](https://ojs.aaai.org/index.php/AAAI/article/view/28690)[[code]](https://lewandofskee.github.io/projects/diad)
 + Exploring Plain ViT Reconstruction for Multi-class Unsupervised Anomaly Detection [[2023]](https://arxiv.org/abs/2312.07495)
 + Structural Teacher-Student Normality Learning for Multi-Class Anomaly Detection and Localization [[2024]](https://arxiv.org/abs/2402.17091)
 + Unsupervised anomaly detection and localization with one model for all category [[KBS 2024]](https://www.sciencedirect.com/science/article/pii/S0950705124001680)
 + Anomaly Detection by Adapting a pre-trained Vision Language Model [[2024]](https://arxiv.org/abs/2403.09493)
 + DMAD: Dual Memory Bank for Real-World Anomaly Detection [[2024]](https://arxiv.org/abs/2403.12362)
 + Toward Multi-class Anomaly Detection: Exploring Class-aware Unified Model against Inter-class Interference [[2024]](https://arxiv.org/abs/2403.14213)
 + Hierarchical Gaussian Mixture Normalizing Flow Modeling for Unified Anomaly Detection [[ECCV 2024]](https://arxiv.org/abs/2403.13349)[[code]](https://github.com/xcyao00/HGAD)
 + Long-Tailed Anomaly Detection with Learnable Class Names [[CVPR 2024]](https://arxiv.org/abs/2403.20236)[[data split]](https://zenodo.org/records/10854201)
 + MambaAD: Exploring State Space Models for Multi-class Unsupervised Anomaly Detection [[NeurIPS 2024]](https://arxiv.org/abs/2404.06564)[[code]](https://lewandofskee.github.io/projects/MambaAD/)
 + Learning Feature Inversion for Multi-class Anomaly Detection under General-purpose COCO-AD Benchmark [[2024]](https://arxiv.org/abs/2404.10760)[[code]](https://github.com/zhangzjn/ader)
 + Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection [[2024]](https://arxiv.org/abs/2405.14325)
 + Prior Normality Prompt Transformer for Multi-class Industrial Image Anomaly Detection [[TII 2024]](https://arxiv.org/abs/2406.11507)
 + An Incremental Unified Framework for Small Defect Inspection [[ECCV2024]](https://arxiv.org/abs/2312.08917v2)[[code]](https://github.com/jqtangust/IUF)
 + Learning Multi-view Anomaly Detection [[2024]](https://arxiv.org/abs/2407.11935)
 + Revitalizing Reconstruction Models for Multi-class Anomaly Detection via Class-Aware Contrastive Learning [[2024]](https://arxiv.org/abs/2412.04769)[[code]](https://github.com/LGC-AD/AD-LGC)

## 3.8 Logical AD
 + Beyond Dents and Scratches: Logical Constraints in Unsupervised Anomaly Detection and Localization [[IJCV 2022]](https://link.springer.com/content/pdf/10.1007/s11263-022-01578-9.pdf)
 + Set Features for Fine-grained Anomaly Detection[[2023]](https://arxiv.org/abs/2302.12245) [[code]](https://github.com/NivC/SINBAD)
 + EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies [[WACV 2024]](https://openaccess.thecvf.com/content/WACV2024/papers/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.pdf)
 + Contextual Affinity Distillation for Image Anomaly Detection [[WACV 2024]](https://openaccess.thecvf.com/content/WACV2024/papers/Zhang_Contextual_Affinity_Distillation_for_Image_Anomaly_Detection_WACV_2024_paper.pdf)
 + REB: Reducing Biases in Representation for Industrial Anomaly Detection [[2023]](https://arxiv.org/abs/2308.12577)[[code]](https://github.com/ShuaiLYU/REB)
 + Learning Global-Local Correspondence with Semantic Bottleneck for Logical Anomaly Detection [[TCSVT 2023]](https://arxiv.org/abs/2303.05768)[[code]](https://github.com/hmyao22/GLCF)
 + Template-guided Hierarchical Feature Restoration for Anomaly Detection [[ICCV 2023]](https://openaccess.thecvf.com/content/ICCV2023/papers/Guo_Template-guided_Hierarchical_Feature_Restoration_for_Anomaly_Detection_ICCV_2023_paper.pdf)
 + Few Shot Part Segmentation Reveals Compositional Logic for Industrial Anomaly Detection [[AAAI 2024]](https://ojs.aaai.org/index.php/AAAI/article/view/28703)[[code]](https://github.com/oopil/PSAD_logical_anomaly_detection)
 + Generating and Reweighting Dense Contrastive Patterns for Unsupervised Anomaly Detection [[AAAI 2024]](https://ojs.aaai.org/index.php/AAAI/article/view/27910)
 + PUAD: Frustratingly Simple Method for Robust Anomaly Detection [[2024]](https://arxiv.org/abs/2402.15143)
 + AnomalyXFusion: Multi-modal Anomaly Synthesis with Diffusion [[2024]](https://arxiv.org/abs/2404.19444)[[data]](https://github.com/hujiecpp/MVTec-Caption)
 + Supervised Anomaly Detection for Complex Industrial Images [[2024]](https://arxiv.org/abs/2405.04953)[[code]](https://github.com/abc-125/segad)
 + SAM-LAD: Segment Anything Model Meets Zero-Shot Logic Anomaly Detection [[2024]](https://arxiv.org/abs/2406.00625)
 + SLSG: Industrial Image Anomaly Detection with Improved Feature Embeddings and One-Class Classification [[PR 2024]](https://www.sciencedirect.com/science/article/pii/S0031320324006137)
 + Unsupervised Component Segmentation for Logical Anomaly Detection [[2024]](https://arxiv.org/abs/2408.15628) [[code]](https://github.com/Tokichan/CSAD)
 + LogiCode: an LLM-Driven Framework for Logical Anomaly Detection [[2024]](https://arxiv.org/pdf/2406.04687)
 + CSAD: Unsupervised Component Segmentation for Logical Anomaly Detection [[BMVC 2024]](https://arxiv.org/abs/2408.15628)[[code]](https://github.com/Tokichan/CSAD)
 + Revisiting Deep Feature Reconstruction for Logical and Structural Industrial Anomaly Detection[[TMLR 2024]](https://arxiv.org/abs/2410.16255)[[code]](https://github.com/sukanyapatra1997/ULSAD-2024)
 + LogicAD: Explainable Anomaly Detection via VLM-based Text Feature Extraction [[2025]](https://arxiv.org/abs/2501.01767)[[code]](https://jasonjin34.github.io/logicad.github.io)

## Other settings
### TTT binary segmentation
+ Test Time Training for Industrial Anomaly Segmentation [[2024]](https://arxiv.org/abs/2404.03743)
### MoE with TTA
+ Adapted-MoE: Mixture of Experts with Test-Time Adaption for Anomaly Detection[[2024]](https://arxiv.org/abs/2409.05611)[[code coming soon]]
### Adversary Attack
+ Adversarially Robust Industrial Anomaly Detection Through Diffusion Model [[2024]](https://arxiv.org/abs/2408.04839)
### Defect Classification
+ AnomalyNCD: Towards Novel Anomaly Class Discovery in Industrial Scenarios [[2024]](https://arxiv.org/abs/2410.14379)[[code coming soon]](https://github.com/HUST-SLOW/AnomalyNCD)
+ MVREC: A General Few-shot Defect Classification Model Using Multi-View Region-Context [[AAAI 2025]](https://arxiv.org/abs/2412.16897)
### Rubustness
+ FiCo: Filter or Compensate: Towards Invariant Representation from Distribution Shift for Anomaly Detection [[AAAI 2025]](https://arxiv.org/abs/2412.10115)[[code]](https://github.com/znchen666/FiCo)

# 4 Dataset
| Dataset                | Class | Normal | Abnormal | Total  | Annotation level  | Source                | Time         |
|------------------------|-------|--------|----------|--------|-------------------|-----------------------|--------------|
| [AITEX](https://www.cvmart.net/dataSets/detail/300)                  | 1     | 140    | 105      | 245    | Segmentation mask | RGB real         | 2019         |
| [Anomaly-ShapeNet](https://github.com/Chopper-233/Anomaly-ShapeNet)       | 40    | -      | -        | 1600   | Point-level mask  | Point-cloud synthetic | CVPR,2024    |
| [BTAD](http://avires.dimi.uniud.it/papers/btad/btad.zip)                   | 3     | -      | -        | 2830   | Segmentation mask | RGB real         | 2021         |
| [CID](https://github.com/LightZH/Insulator-Defect-Detection) | 1 | 4060 | 233 | 4293 | Segmentation mask | RGB real | 2024,TIM | 
| [DAGM](https://www.kaggle.com/datasets/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection)                   | 10    | -      | -        | 11500  | Segmentation mask | RGB synthetic     | 2007         |
| [DEEPPCB](https://github.com/tangsanli5201/DeepPCB)                | 1     | -      | -        | 1500   | Bounding box      | RGB synthetic     | 2019         |
| [DTD-Synthetic](https://drive.google.com/drive/folders/10OyPzvI3H6llCZBxKxFlKWt1Pw1tkMK1)          | 12    | -      | -        | -      | Segmentation mask | RGB synthetic     | WACV,2024    |
| [Eyecandies](https://eyecan-ai.github.io/eyecandies/)             | 10    | 13250  | 2250     | 15500  | Segmentation mask | RGBD synthetic image  | ACCV,2022    |
| [Fabirc dataset](http://hub.hku.hk/bitstream/10722/229176/1/content.pdf)         | 1     | 25     | 25       | 50     | Segmentation mask | RGB synthetic     | PR,2016      |
| [GDXray](https://domingomery.ing.puc.cl/material/gdxray/)                 | 1     | 0      | 19407    | 19407  | Bounding box      | RGB real         | 2016         |
| [IPAD](https://ljf1113.github.io/IPAD_VAD/) | 16 | - | - | 597979 | Image | Video real&synthetic | 2024 |
| [KolekrotSDD](https://www.vicos.si/resources/kolektorsdd/)            | 1     | 347    | 52       | 399    | Segmentation mask | RGB real         | JIM,2019     |
| [KolekrotSDD2](https://www.vicos.si/resources/kolektorsdd2/)           | 1     | 2979   | 356      | 3335   | Segmentation mask | RGB real         | CiI,2021     |
| [MIAD](https://miad-2022.github.io/)                   | 7     | 87500  | 17500    | 105000 | Segmentation mask | RGB synthetic     | 2023         |
| [MPDD](https://github.com/stepanje/MPDD)                   | 6     | 1064   | 282      | 1346   | Segmentation mask | RGB real         | ICUMT,2021   |
| [MTD](https://github.com/abin24/Magnetic-tile-defect-datasets.) | 1 | 952 | 392 | 1344 | Segmentation mask | RGB real | CASE,2018 |
| [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)               | 15    | 4096   | 1258     | 5354   | Segmentation mask | RGB real         | CVPR,2019    |
| [MVTec 3D-AD](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad)            | 10    | 2904   | 948      | 3852   | Segmentation mask | RGB real          | VISAPP,2021  |
| [MVTec LOCO-AD](https://www.mvtec.com/company/research/datasets/mvtec-loco)          | 5     | 2347   | 993      | 3340   | Segmentation mask | RGBD real       | IJCV,2022    |
| [NanoTwice](http://web.mi.imati.cnr.it/ettore/NanoTwice)              | 1     | 5      | 40       | 45     | Segmentation mask | RGB real         | TII,2016     |
| [NEU surface defect](http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm)     | 1     | 0      | 1800     | 1800   | Bounding box      | RGB real         | 2013         |
| [PAD](https://github.com/EricLee0224/PAD)                    | 20    | 5231   | 4902     | 10133  | Segmentation mask | RBG synthetic    | NeurIPS,2023 |
| [Real-IAD](https://realiad4ad.github.io/Real-IAD/)               | 30    | 99721  | 51329    | 151050 | Segmentation mask | RGB real         | CVPR,2024    |
| [Real3D-AD](https://github.com/M-3LAB/Real3D-AD)              | 12    | 652    | 602      | 1254   | Point-level mask  | Point-cloud real       | NeurIPS,2023 |
| [RSDD](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8063875)                   | 2     | -      | -        | 195    | Segmentation mask | RGB real         | 2017         |
| [Steel defect detection](https://www.kaggle.com/code/ekhtiar/resunet-a-baseline-on-tensorflow/notebook) | 1     | -      | -        | 18076  | Image             | RGB real         | 2019         |
| [Steel tube dataset](https://github.com/huangyebiaoke/steel-pipe-weld-defect-detection)     | 1     | 0      | 3408     | 3408   | Bounding box      | RGB real         | 2021         |
| [VisA](https://github.com/amazon-science/spot-diff)                   | 12    | 9621   | 1200     | 10821  | Segmentation mask | RGB real         | ECCV,2022    |
| [RAD](https://github.com/hustCYQ/RAD-dataset)                   | 4    | 213   | 1224     | 1224  | Segmentation mask | RGB real         | CASE,2024    |


 + (NEU surface defect dataset)A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects [[2013]](https://www.sciencedirect.com/science/article/pii/S0169433213016437/pdfft?md5=478bf7f07bbf551a5d991048f9bc16e4&pid=1-s2.0-S0169433213016437-main.pdf) [[data]](http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm)
 + (Steel tube dataset)Deep learning based steel pipe weld defect detection [[2021]](https://www.tandfonline.com/doi/pdf/10.1080/08839514.2021.1975391?needAccess=true) [[data]](https://github.com/huangyebiaoke/steel-pipe-weld-defect-detection)
 + (Steel defect dataset)Severstal: Steel Defect Detection [[data 2019]](https://www.kaggle.com/code/ekhtiar/resunet-a-baseline-on-tensorflow/notebook)
 + (NanoTwice)Defect detection in SEM images of nanofibrous materials [[TII 2016]](https://re.public.polimi.it/bitstream/11311/1024586/1/anomaly_detection_sem.pdf) [[data]](http://web.mi.imati.cnr.it/ettore/NanoTwice)
 + (GDXray)GDXray: The database of X-ray images for nondestructive testing [[2015]](http://dmery.sitios.ing.uc.cl/Prints/ISI-Journals/2015-JNDE-GDXray.pdf) [[data]](https://domingomery.ing.puc.cl/material/gdxray/)
 + (DEEP PCB)Online PCB defect detector on a new PCB defect dataset [[2019]](https://arxiv.org/pdf/1902.06197.pdf) [[data]](https://github.com/tangsanli5201/DeepPCB)
 + (PCBA-defect) A PCB Dataset for Defects Detection and Classification [[2019]](https://arxiv.org/abs/1901.08204)[[data]](https://www.kaggle.com/datasets/akhatova/pcb-defects)
 + (CPLID) Insulator Data Set - Chinese Power Line Insulator Dataset [[data]](https://github.com/InsulatorData/InsulatorDataSet)
 + (Fabric dataset)Fabric inspection based on the Elo rating method [[PR 2016]](http://hub.hku.hk/bitstream/10722/229176/1/content.pdf)
 + (KolektorSDD)Segmentation-based deep-learning approach for surface-defect detection [[Journal of Intelligent Manufacturing]](http://arxiv.org/pdf/1903.08536) [[data]](https://www.vicos.si/resources/kolektorsdd/)
 + (KolektorSDD2)Mixed supervision for surface-defect detection: From weakly to fully supervised learning [[Computers in Industry 2021]](https://arxiv.org/pdf/2104.06064.pdf) [[data]](https://www.vicos.si/resources/kolektorsdd2/)
 + SensumSODF-dataset: Detection of surface defects on pharmaceutical solid oral dosage forms with convolutional neural networks[[Neural Computing and Applications 2021]](https://link.springer.com/article/10.1007/s00521-021-06397-6)[[data]](https://www.sensum.eu/sensumsodf-dataset/)
 + (RSDD)A hierarchical extractor-based visual rail surface inspection system [[2017]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8063875)
 + (Eyecandies)The Eyecandies Dataset for Unsupervised Multimodal Anomaly Detection and Localization [[ACCV 2022]](https://arxiv.org/pdf/2210.04570.pdf) [[data]](https://eyecan-ai.github.io/eyecandies/)
 + (MVTec AD)MVTec AD: A comprehensive real-world dataset for unsupervised anomaly detection [[CVPR 2019]](https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.html) [[IJCV 2021]](https://link.springer.com/content/pdf/10.1007/s11263-020-01400-4.pdf) [[data]](https://www.mvtec.com/company/research/datasets/mvtec-ad)✨✨✨
 + (MVTec 3D-AD)The mvtec 3d-ad dataset for unsupervised 3d anomaly detection and localization [[VISAPP 2021]](https://arxiv.org/pdf/2112.09045.pdf) [[data]](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad)✨✨
 + (MVTec LOCO-AD)Beyond Dents and Scratches: Logical Constraints in Unsupervised Anomaly Detection and Localization [[IJCV 2022]](https://link.springer.com/content/pdf/10.1007/s11263-022-01578-9.pdf) [[data]](https://www.mvtec.com/company/research/datasets/mvtec-loco)✨✨✨
 + (MPDD)Deep learning-based defect detection of metal parts: evaluating current methods in complex conditions [[ICUMT 2021]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9631567) [[data]](https://github.com/stepanje/MPDD)
 + (MPDD2)Anomaly detection for real-world industrial applications: benchmarking recent self-supervised and pretrained methods [[ICUMT 2022]](https://ieeexplore.ieee.org/abstract/document/9943437) [[data]](https://github.com/stepanje/MPDD2)
 + (BTAD)VT-ADL: A vision transformer network for image anomaly detection and localization [[2021]](http://arxiv.org/pdf/2104.10036) [[data]](http://avires.dimi.uniud.it/papers/btad/btad.zip)
 + (VisA)SPot-the-Difference Self-supervised Pre-training for Anomaly Detection and Segmentation [[ECCV 2022]](https://arxiv.org/pdf/2207.14315.pdf) [[data]](https://github.com/amazon-science/spot-diff)✨✨✨
 + (MTD)Surface defect saliency of magnetic tile [[2020]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8560423) [[data]](https://github.com/abin24/Magnetic-tile-defect-datasets.)
 + (DAGM)DAGM dataset [[data 2007]](https://www.kaggle.com/datasets/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection)
 + (MIAD)Miad:A maintenance inspection dataset for unsupervised anomaly detection [[2022]](https://arxiv.org/abs/2211.13968) [[data]](https://miad-2022.github.io/)✨✨
 + CVPR 1st workshop on Vision-based InduStrial InspectiON [[homepage]](https://vision-based-industrial-inspection.github.io/cvpr-2023/) [[data]](https://drive.google.com/drive/folders/1TVp_UXJuXudqhC2L3ZKyIDcmQ_2O3JVi)
 + (SSGD)SSGD: A smartphone screen glass dataset for defect detection [[2023]](https://arxiv.org/abs/2303.06673)[[github page]](https://github.com/VincentHancoder/SSGD)
 + (AeBAD)Industrial Anomaly Detection with Domain Shift: A Real-world Dataset and Masked Multi-scale Reconstruction [[2023]](https://arxiv.org/abs/2304.02216) [[data]](https://github.com/zhangzilongc/MMR)
 + VISION Datasets: A Benchmark for Vision-based InduStrial InspectiON [[2023]](https://arxiv.org/abs/2306.07890) [[data]](https://huggingface.co/datasets/VISION-Workshop/VISION-Datasets)✨✨✨
 + PAD: A Dataset and Benchmark for Pose-agnostic Anomaly Detection [[NeurIPS 2023]](https://github.com/EricLee0224/PAD)
 + PKU-GoodsAD: A Supermarket Goods Dataset for Unsupervised Anomaly Detection and Segmentation [[2023]](https://github.com/jianzhang96/GoodsAD)[[data]](https://github.com/jianzhang96/GoodsAD)✨✨
 + Real3D-AD: A Dataset of Point Cloud Anomaly Detection [[NeurIPS 2023]](https://openreview.net/pdf?id=zGthDp4yYe)[[data]](https://github.com/M-3LAB/Real3D-AD)✨✨✨
 + InsPLAD: A Dataset and Benchmark for Power Line Asset Inspection in UAV Images [[IJRS 2023]](https://arxiv.org/abs/2311.01619)[[data]](https://github.com/andreluizbvs/InsPLAD)
 + Image-Pointcloud Fusion based Anomaly Detection using PD-REAL Dataset [[2023]](https://arxiv.org/abs/2311.04095)[[data]](https://github.com/Andy-cs008/PD-REAL)
 + CrashCar101: Procedural Generation for Damage Assessment [[WACV 2024]](https://crashcar.compute.dtu.dk/static/2435.pdf)[[data]](https://crashcar.compute.dtu.dk/)
 + Defect Spectrum: A Granular Look of Large-Scale Defect Datasets with Rich Semantics [[ECCV 2024]](https://openreview.net/forum?id=RLhS1TrjK3)[[data]](https://github.com/EnVision-Research/Defect_Spectrum)
 + (DTD-Synthetic) Zero-shot versus Many-shot: Unsupervised Texture Anomaly Detection [[WACV 2023]](https://ieeexplore.ieee.org/document/10030870)[[data]](https://drive.google.com/drive/folders/10OyPzvI3H6llCZBxKxFlKWt1Pw1tkMK1)
 + Towards Scalable 3D Anomaly Detection and Localization: A Benchmark via 3D Anomaly Synthesis and A Self-Supervised Learning Network [[CVPR 2024]](https://arxiv.org/abs/2311.14897)[[data]](https://github.com/Chopper-233/Anomaly-ShapeNet)
 + Real-IAD: A Real-World Multi-view Dataset for Benchmarking Versatile Industrial Anomaly Detection [[CVPR 2024]](https://arxiv.org/abs/2403.12580)[[code]](https://github.com/Tencent/AnomalyDetection_Real-IAD)[[data]](https://realiad4ad.github.io/Real-IAD/)✨✨✨
 + Catenary Insulator Defects Detection: A Dataset and an Unsupervised Baseline [[TIM 2024]](https://ieeexplore.ieee.org/abstract/document/10504848)[[code]](https://github.com/LightZH/Insulator-Defect-Detection)
 + IPAD: Industrial Process Anomaly Detection Dataset [[2024]](https://arxiv.org/abs/2404.15033)[[data]](https://ljf1113.github.io/IPAD_VAD/)
 + MVTec-Caption: AnomalyXFusion: Multi-modal Anomaly Synthesis with Diffusion [[2024]](https://arxiv.org/abs/2404.19444)[[data]](https://github.com/hujiecpp/MVTec-Caption)
 + Supervised Anomaly Detection for Complex Industrial Images [[2024]](https://arxiv.org/abs/2405.04953)[[data]](https://github.com/abc-125/segad)
 + PeanutAD: A Real-World Dataset for Anomaly Detection in Agricultural Product Processing Line [[2024]](https://ieeexplore.ieee.org/abstract/document/10634679)[[data]](https://github.com/TCV0257/PeanutAD)
 + The Woven Fabric Defect Detection (WFDD) dataset [[2024]](https://arxiv.org/abs/2407.09359)[[data]](https://github.com/cqylunlun/GLASS?tab=readme-ov-file#1wfdd-download-link)
 + Texture-AD: An Anomaly Detection Dataset and Benchmark for Real Algorithm Development[[2024]](https://arxiv.org/abs/2409.06367)[[data]](https://huggingface.co/datasets/texture-ad/Texture-AD-Benchmark)
 + MulSen-AD: A Dataset and Benchmark for Multi-Sensor Anomaly Detection [[2024]](https://zzzbbbzzz.github.io/MulSen_AD/index.html)[[data]](https://github.com/ZZZBBBZZZ/MulSen-AD)
 + CableInspect-AD: An Expert-Annotated Anomaly Detection Dataset [[NeurIPS 2024]](https://arxiv.org/abs/2409.20353)[[data]](https://mila-iqia.github.io/cableinspect-ad/)
 + RAD: A Dataset and Benchmark for Real-Life Anomaly Detection with Robotic Observations [[2024]](https://arxiv.org/html/2410.00713v1)[[data]](https://github.com/kaichen-z/RAD)
 + AD3: Introducing a score for Anomaly Detection Dataset Difficulty assessment using VIADUCT dataset [[ECCV 2024]](https://eccv.ecva.net/virtual/2024/poster/2287)
 + MMAD: The First-Ever Comprehensive Benchmark for Multimodal Large Language Models in Industrial Anomaly Detection [[2024]](https://arxiv.org/abs/2410.09453) [[data]](https://github.com/jam-cc/MMAD)
 + MANTA: A Large-Scale Multi-View and Visual-Text Anomaly Detection Dataset for Tiny Objects [[2024]](https://arxiv.org/abs/2412.04867)[[data]](https://grainnet.github.io/MANTA)
 + Are Anomaly Scores Telling the Whole Story? A Benchmark for Multilevel Anomaly Detection [[2024]](https://arxiv.org/abs/2411.14515)


## BibTex Citation

If you find this paper and repository useful, please cite our paper☺️.

```
@article{liu2024deep,
  title={Deep industrial image anomaly detection: A survey},
  author={Liu, Jiaqi and Xie, Guoyang and Wang, Jinbao and Li, Shangnian and Wang, Chengjie and Zheng, Feng and Jin, Yaochu},
  journal={Machine Intelligence Research},
  volume={21},
  number={1},
  pages={104--135},
  year={2024},
  publisher={Springer}
}

@article{xie2024iad,
  title={Im-iad: Industrial image anomaly detection benchmark in manufacturing},
  author={Xie, Guoyang and Wang, Jinbao and Liu, Jiaqi and Lyu, Jiayi and Liu, Yong and Wang, Chengjie and Zheng, Feng and Jin, Yaochu},
  journal={IEEE Transactions on Cybernetics},
  year={2024},
  publisher={IEEE}
}

@article{jiang2022survey,
  title={A survey of visual sensory anomaly detection},
  author={Jiang, Xi and Xie, Guoyang and Wang, Jinbao and Liu, Yong and Wang, Chengjie and Zheng, Feng and Jin, Yaochu},
  journal={arXiv preprint arXiv:2202.07006},
  year={2022}
}
```
## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=M-3LAB/awesome-industrial-anomaly-detection&type=Date)](https://star-history.com/#M-3LAB/awesome-industrial-anomaly-detection&Date)
