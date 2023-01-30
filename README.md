# Awesome Industrial Anomaly Detection
We will keep focusing on this field and update relevant information.

[Deep Visual Anomaly Detection in Industrial Manufacturing: A Survey](https://arxiv.org/abs/2301.11514)

# Paper Tree
![](https://github.com/M-3LAB/awesome-industrial-anomaly-detection/blob/main/paper_tree.png)
# Timeline
![](https://github.com/M-3LAB/awesome-industrial-anomaly-detection/blob/main/timeline.png)

Paper list for industrial image anomaly detection
# 1 Introduction
+ (yang2021generalized)Generalized out-of-distribution detection: A survey [[2021]](https://arxiv.org/pdf/2110.11334.pdf)
+ (bergmann2019mvtec)MVTec AD: A comprehensive real-world dataset for unsupervised anomaly detection [[2019]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.pdf)
+ (bergmann2021mvtec)The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection [[2021]](https://link.springer.com/content/pdf/10.1007/s11263-020-01400-4.pdf)
+ (czimmermann2020visual)Visual-based defect detection and classification approaches for industrial applications: a survey [[2020]](https://pdfs.semanticscholar.org/1dfc/080a5f26b5ce78f9ce3e9f106bf7e8124f74.pdf)
+ (tao2022deep)Deep Learning for Unsupervised Anomaly Localization in Industrial Images: A Survey [[2022]](http://arxiv.org/pdf/2207.10298)
+ (cui2022survey)A Survey on Unsupervised Industrial Anomaly Detection Algorithms [[2022]](https://arxiv.org/abs/2204.11161)
+ A review on computer vision based defect detection and condition assessment of concrete and asphalt civil infrastructure [[2015]](https://www.sciencedirect.com/science/article/abs/pii/S1474034615000208)

# 2 Unsupervised AD

## 2.1 Feature-Embedding-based Methods

### 2.1.1 Teacher-Student
+ (bergmann2020uninformed)Uninformed students: Student-teacher anomaly detection with discriminative latent embeddings [[2020]](http://arxiv.org/pdf/1911.02357)
+ (Wang2021StudentTeacherFP)Student-Teacher Feature Pyramid Matching for Anomaly Detection [[2021]](https://arxiv.org/pdf/2103.04257.pdf)
+ (salehi2021multiresolution)Multiresolution knowledge distillation for anomaly detection [[2020]](https://arxiv.org/pdf/2011.11108)
+ (yamada2021reconstruction)Reconstruction Student with Attention for Student-Teacher Pyramid Matching [[2021]](https://arxiv.org/pdf/2111.15376.pdf)
+ (yamada2022reconstructed)Reconstructed Student-Teacher and Discriminative Networks for Anomaly Detection [[2022]](https://arxiv.org/pdf/2210.07548.pdf)
+ (deng2022anomaly)Anomaly Detection via Reverse Distillation from One-Class Embedding [[2022]](http://arxiv.org/pdf/2201.10703)
+ (rudolph2022asymmetric)Asymmetric Student-Teacher Networks for Industrial Anomaly Detection [[2022]](https://arxiv.org/pdf/2210.07829.pdf)
+ (cao2022informative)Informative knowledge distillation for image anomaly segmentation [[2022]](https://www.sciencedirect.com/science/article/pii/S0950705122004038/pdfft?md5=758c327dd4d1d052b61a19882f957123&pid=1-s2.0-S0950705122004038-main.pdf)

### 2.1.2 One-Class Classification (OCC)
+ (yi2020patch)Patch svdd: Patch-level svdd for anomaly detection and segmentation [[2020]](https://arxiv.org/pdf/2006.16067.pdf)
+ (zhang2021anomaly)Anomaly detection using improved deep SVDD model with data structure preservation [[2021]](https://www.sciencedirect.com/science/article/am/pii/S0167865521001598)
+ (hu2021semantic)A Semantic-Enhanced Method Based On Deep SVDD for Pixel-Wise Anomaly Detection [[2021]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9428370)
+ (massoli2021mocca)MOCCA: Multilayer One-Class Classification for Anomaly Detection [[2021]](http://arxiv.org/pdf/2012.12111)
+ (sauter2021defect)Defect Detection of Metal Nuts Applying Convolutional Neural Networks [[2021]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9529439)
+ (reiss2021panda)Panda: Adapting pretrained features for anomaly detection and segmentation [[2021]](http://arxiv.org/pdf/2010.05903)
+ (reiss2021mean)Mean-shifted contrastive loss for anomaly detection [[2021]](https://arxiv.org/pdf/2106.03844.pdf)
+ (sohn2020learning)Learning and Evaluating Representations for Deep One-Class Classification [[2020]](https://arxiv.org/pdf/2011.02578.pdf)
+ (yoa2021self)Self-supervised learning for anomaly detection with dynamic local augmentation [[2021]](https://ieeexplore.ieee.org/ielx7/6287639/6514899/09597511.pdf)
+ (de2021contrastive)Contrastive Predictive Coding for Anomaly Detection [[2021]](https://arxiv.org/pdf/2107.07820.pdf)
+ (li2021cutpaste)Cutpaste: Self-supervised learning for anomaly detection and localization [[2021]](http://arxiv.org/pdf/2104.04015)
+ (iquebal2020consistent)Consistent estimation of the max-flow problem: Towards unsupervised image segmentation [[2020]](http://arxiv.org/pdf/1811.00220)
+ (yang2022memseg)MemSeg: A semi-supervised method for image surface defect detection using differences and commonalities [[2022]](https://arxiv.org/pdf/2205.00908.pdf)

### 2.1.3 Distribution-Map
+ (tailanian2021multi)A Multi-Scale A Contrario method for Unsupervised Image Anomaly Detection [[2021]](http://arxiv.org/pdf/2110.02407)
+ (rippel2021modeling)Modeling the distribution of normal data in pre-trained deep features for anomaly detection [[2021]](http://arxiv.org/pdf/2005.14140)
+ (rippel2021transfer)Transfer Learning Gaussian Anomaly Detection by Fine-Tuning Representations [[2021]](https://arxiv.org/pdf/2108.04116.pdf)
+ (zhang2022pedenet)PEDENet: Image anomaly localization via patch embedding and density estimation [[2022]](https://arxiv.org/pdf/2110.15525.pdf)
+ (wan2022unsupervised)Unsupervised image anomaly detection and segmentation based on pre-trained feature mapping [[2022]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9795121)
+ (wan2022position)Position Encoding Enhanced Feature Mapping for Image Anomaly Detection [[2022]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9926547)
+ (zheng2022focus)Focus your distribution: Coarse-to-fine non-contrastive learning for anomaly detection and localization [[2022]](http://arxiv.org/pdf/2110.04538)
+ (yu2021fastflow)Fastflow: Unsupervised anomaly detection and localization via 2d normalizing flows [[2021]](https://arxiv.org/pdf/2111.07677.pdf)
+ (rudolph2021same)Same same but differnet: Semi-supervised defect detection with normalizing flows [[2021]](http://arxiv.org/pdf/2008.12577)
+ (rudolph2022fully)Fully convolutional cross-scale-flows for image-based defect detection [[2022]](http://arxiv.org/pdf/2110.02855)
+ (gudovskiy2022cflow)Cflow-ad: Real-time unsupervised anomaly detection with localization via conditional normalizing flows [[2022]](http://arxiv.org/pdf/2107.12571)
+ (yan2022cainnflow)CAINNFlow: Convolutional block Attention modules and Invertible Neural Networks Flow for anomaly detection and localization tasks [[2022]](https://arxiv.org/pdf/2206.01992.pdf)
+ (kim2022altub)AltUB: Alternating Training Method to Update Base Distribution of Normalizing Flow for Anomaly Detection [[2022]](https://arxiv.org/pdf/2210.14913.pdf)

### 2.1.4 Memory Bank
 <!-- + (eskin2002geometric) A geometric framework for unsupervised anomaly detection [[2002]](https://www.researchgate.net/publication/242620986_A_Geometric_Framework_for_Unsupervised_Anomaly_Detection_Detecting_Intrusions_in_Unlabeled_Data#read) -->
 + (cohen2020sub)Sub-image anomaly detection with deep pyramid correspondences [[2020]](https://arxiv.org/pdf/2005.02357.pdf)
 + (kim2021semi)Semi-orthogonal embedding for efficient unsupervised anomaly segmentation [[2021]](https://arxiv.org/pdf/2105.14737.pdf)
 + (li2021anomaly)Anomaly Detection Via Self-Organizing Map [[2021]](http://arxiv.org/pdf/2107.09903)
 + (wan2021industrial)Industrial Image Anomaly Localization Based on Gaussian Clustering of Pretrained Feature [[2021]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9479740)
 + (roth2022towards)Towards total recall in industrial anomaly detection[[2022]](http://arxiv.org/pdf/2106.08265)
 + (Lee2022CFACF)CFA: Coupled-Hypersphere-Based Feature Adaptation for Target-Oriented Anomaly Localization[[2022]](https://arxiv.org/pdf/2206.04325.pdf)
 + (kim2022fapm)FAPM: Fast Adaptive Patch Memory for Real-time Industrial Anomaly Detection[[2022]](https://arxiv.org/pdf/2211.07381.pdf)
 + (jang2022n)N-pad: Neighboring Pixel-based Industrial Anomaly Detection [[2022]](https://arxiv.org/pdf/2210.08768.pdf)
 + (bae2022image)Image Anomaly Detection and Localization with Position and Neighborhood Information [[2022]](https://arxiv.org/pdf/2211.12634.pdf)
 + (tsai2022multi)Multi-scale patch-based representation learning for image anomaly detection and segmentation [[2022]](https://openaccess.thecvf.com/content/WACV2022/papers/Tsai_Multi-Scale_Patch-Based_Representation_Learning_for_Image_Anomaly_Detection_and_Segmentation_WACV_2022_paper.pdf)
 + (zou2022spot)SPot-the-Difference Self-supervised Pre-training for Anomaly Detection and Segmentation [[2022]](https://arxiv.org/pdf/2207.14315.pdf)
 

## 2.2 Reconstruction-Based Methods

### 2.2.1 Autoencoder (AE)
 + (bergmann2018improving)Improving unsupervised defect segmentation by applying structural similarity to autoencoders [[2018]](https://arxiv.org/pdf/1807.02011.pdf)
 + (chung2020unsupervised)Unsupervised anomaly detection using style distillation [[2020]](https://ieeexplore.ieee.org/ielx7/6287639/6514899/09288772.pdf)
 + (liu2021unsupervised)Unsupervised two-stage anomaly detection [[2021]](https://arxiv.org/pdf/2103.11671.pdf)
 + (yang2020dfr)Dfr: Deep feature reconstruction for unsupervised anomaly segmentation [[2020]](https://arxiv.org/pdf/2012.07122.pdf)
 + (yan2021unsupervised)Unsupervised anomaly segmentation via multilevel image reconstruction and adaptive attention-level transition [[2021]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9521893)
 + (zhou2020encoding)Encoding structure-texture relation with p-net for anomaly detection in retinal images [[2020]](http://arxiv.org/pdf/2008.03632)
 + (collin2021improved)Improved anomaly detection by training an autoencoder with skip connections on images corrupted with stain-shaped noise [[2021]](http://arxiv.org/pdf/2008.12977)
 + (tao2022unsupervised)Unsupervised anomaly detection for surface defects with dual-siamese network [[2022]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9681338)
 + (hou2021divide)Divide-and-assemble: Learning block-wise memory for unsupervised anomaly detection [[2021]](https://openaccess.thecvf.com/content/ICCV2021/papers/Hou_Divide-and-Assemble_Learning_Block-Wise_Memory_for_Unsupervised_Anomaly_Detection_ICCV_2021_paper.pdf)
 + (liu2022reconstruction)Reconstruction from edge image combined with color and gradient difference for industrial surface anomaly detection [[2022]](http://arxiv.org/pdf/2210.14485)
 + (kim2022spatial)Spatial Contrastive Learning for Anomaly Detection and Localization [[2022]](https://ieeexplore.ieee.org/ielx7/6287639/9668973/09709224.pdf)
 + (li2020superpixel)Superpixel masking and inpainting for self-supervised anomaly detection [[2020]](https://www.bmvc2020-conference.com/assets/papers/0275.pdf)
 + (nakanishi2020iterative)Iterative image inpainting with structural similarity mask for anomaly detection [[2020]](https://openreview.net/pdf?id=b4ach0lGuYO)
 + (huang2022self)Self-Supervised Masking for Unsupervised Anomaly Detection and Localization [[2022]](https://arxiv.org/pdf/2205.06568.pdf)
 + (zavrtanik2021reconstruction)Reconstruction by inpainting for visual anomaly detection [[2021]](https://www.sciencedirect.com/science/article/pii/S0031320320305094/pdfft?md5=9bbe942017de1acd3a97034bc2d4a8fb&pid=1-s2.0-S0031320320305094-main.pdf)
 + (zavrtanik2021draem)Draem-a discriminatively trained reconstruction embedding for surface anomaly detection [[2021]](http://arxiv.org/pdf/2108.07610)
 + (zavrtanik2022dsr)DSR: A dual subspace re-projection network for surface anomaly detection [[2022]](https://arxiv.org/pdf/2208.01521.pdf)
 + (schluter2022natural)Natural Synthetic Anomalies for Self-supervised Anomaly Detection and Localization [[2022]](https://arxiv.org/pdf/2109.15222.pdf)
 + (bauer2022self)Self-Supervised Training with Autoencoders for Visual Anomaly Detection [[2022]](https://arxiv.org/pdf/2206.11723.pdf)
 + (ristea2022self)Self-supervised predictive convolutional attentive block for anomaly detection [[2022]](http://arxiv.org/pdf/2111.09099)
 + (madan2022self)Self-Supervised Masked Convolutional Transformer Block for Anomaly Detection [[2022]](https://arxiv.org/pdf/2209.12148.pdf)
 + (dehaene2019iterative)Iterative energy-based projection on a normal data manifold for anomaly localization [[2019]](https://arxiv.org/pdf/2002.03734.pdf)
 + (liu2020towards)Towards visually explaining variational autoencoders [[2020]](http://arxiv.org/pdf/1911.07389)
 + (matsubara2020deep)Deep generative model using unregularized score for anomaly detection with heterogeneous complexity [[2020]](http://arxiv.org/pdf/1807.05800)
 + (dehaene2020anomaly)Anomaly localization by modeling perceptual features [[2020]](https://arxiv.org/pdf/2008.05369.pdf)
 + (wang2020image)Image anomaly detection using normal data only by latent space resampling [[2020]](https://pdfs.semanticscholar.org/cb59/dab0a725c0b511f3140ea47ea0967f3643bf.pdf)

### 2.2.2 Generative Adversarial Networks (GANs)
 + (yan2021learning)Learning semantic context from normal samples for unsupervised anomaly detection [[2021]](https://ojs.aaai.org/index.php/AAAI/article/download/16420/16227)
 + (song2021anoseg)Anoseg: Anomaly segmentation network using self-supervised learning [[2021]](https://arxiv.org/pdf/2110.03396.pdf)
 + (liang2022omni)Omni-frequency Channel-selection Representations for Unsupervised Anomaly Detection [[2022]](https://arxiv.org/pdf/2203.00259.pdf)

### 2.2.3 Transformer
 + (mishra2021vt)VT-ADL: A vision transformer network for image anomaly detection and localization [[2021]](http://arxiv.org/pdf/2104.10036)
 + (you2022adtr)ADTR: Anomaly Detection Transformer with Feature Reconstruction [[2022]](https://arxiv.org/pdf/2209.01816.pdf)
 + (lee2022anovit)AnoViT: Unsupervised Anomaly Detection and Localization With Vision Transformer-Based Encoder-Decoder [[2022]](https://ieeexplore.ieee.org/ielx7/6287639/6514899/09765986.pdf)
 + (mathian2022haloae)HaloAE: An HaloNet based Local Transformer Auto-Encoder for Anomaly Detection and Localization [[2022]](https://arxiv.org/pdf/2208.03486.pdf)
 + (pirnay2022inpainting)Inpainting transformer for anomaly detection [[2022]](https://arxiv.org/pdf/2104.13897.pdf)
 + (jiang2022masked)Masked Swin Transformer Unet for Industrial Anomaly Detection [[2022]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9858596)
 + (de2022masked)Masked Transformer for image Anomaly Localization [[2022]](http://arxiv.org/pdf/2210.15540)

### 2.2.4 Diffusion Model
 + (ho2020denoising)Denoising diffusion probabilistic models [[2020]](https://arxiv.org/pdf/2006.11239.pdf)
 + (wyatt2022anoddpm)AnoDDPM: Anomaly Detection With Denoising Diffusion Probabilistic Models Using Simplex Noise [[2022]](http://dro.dur.ac.uk/36134/1/36134.pdf)
 + (teng2022unsupervised)Unsupervised Visual Defect Detection with Score-Based Generative Model[[2022]](https://arxiv.org/pdf/2211.16092.pdf)
 # 2.3 Supervised AD
 + (chu2020neural)Neural batch sampling with reinforcement learning for semi-supervised anomaly detection [[2020]](https://www.ri.cmu.edu/wp-content/uploads/2020/05/WenHsuan_MSR_Thesis-1.pdf)
 + (liznerski2020explainable)Explainable Deep One-Class Classification [[2020]](https://arxiv.org/pdf/2007.01760.pdf)
 + (venkataramanan2020attention)Attention guided anomaly localization in images [[2020]](http://arxiv.org/pdf/1911.08616)
 + (bovzivc2021mixed)Mixed supervision for surface-defect detection: From weakly to fully supervised learning [[2021]](https://arxiv.org/pdf/2104.06064.pdf)
 + (pang2021explainable)Explainable deep few-shot anomaly detection with deviation networks [[2021]](https://arxiv.org/pdf/2108.00462.pdf)
 + (wan2022logit)Logit Inducing With Abnormality Capturing for Semi-Supervised Image Anomaly Detection [2022]
 + (ding2022catching)Catching Both Gray and Black Swans: Open-set Supervised Anomaly Detection [[2022]](http://arxiv.org/pdf/2203.14506)
 + (sindagi2017domain)Domain adaptation for automatic OLED panel defect detection using adaptive support vector data description [[2017]](https://link.springer.com/article/10.1007/s11263-016-0953-y)
 + (qiu2021effective)An effective framework of automated visual surface defect detection for metal parts [[2021]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9475966)
 + (bhattacharya2021interleaved)Interleaved Deep Artifacts-Aware Attention Mechanism for Concrete Structural Defect Classification [[2021]](https://eprints.keele.ac.uk/10031/1/TIP24Jul2021.pdf)
 + (zeng2021reference)Reference-based defect detection network [[2021]](http://arxiv.org/pdf/2108.04456)
 + (long2021fabric)Fabric defect detection using tactile information [[2021]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9561092)
 + (hu2020lightweight)A lightweight spatial and temporal multi-feature fusion network for defect detection [[2020]](http://nrl.northumbria.ac.uk/id/eprint/48908/1/ALightweightSpatialandTemporalMulti-featureFusionNetworkforDefectDetection.pdf)
 + (ferguson2018detection)Detection and segmentation of manufacturing defects with convolutional neural networks and transfer learning [[2018]](https://europepmc.org/articles/pmc6512995?pdf=render)

# 3 Other Research Direction

## 3.1 Few-Shot AD
 + (wu2021learning)Learning unsupervised metaformer for anomaly detection [[2021]](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_Learning_Unsupervised_Metaformer_for_Anomaly_Detection_ICCV_2021_paper.pdf)
 + (huang2022registration)Registration based few-shot anomaly detection [[2022]](https://arxiv.org/pdf/2207.07361.pdf)
 + (rudolph2021same)Same same but differnet: Semi-supervised defect detection with normalizing flows [[2021]](http://arxiv.org/pdf/2008.12577)
 + (roth2022towards)Towards total recall in industrial anomaly detection [[2022]](http://arxiv.org/pdf/2106.08265)
 + (sheynin2021hierarchical)A hierarchical transformation-discriminating generative model for few shot anomaly detection [[2021]](http://arxiv.org/pdf/2104.14535)
 + (kamoona2021anomaly)Anomaly detection of defect using energy of point pattern features within random finite set framework [[2021]](https://arxiv.org/pdf/2108.12159.pdf)
 + (schwartz2022maeday)MAEDAY: MAE for few and zero shot AnomalY-Detection [[2022]](https://arxiv.org/pdf/2211.14307.pdf)
  <!-- + (he2022masked)Masked autoencoders are scalable vision learners [[2022]](http://arxiv.org/pdf/2111.06377) -->

## 3.2 Noisy AD
 + (tan2021trustmae)Trustmae: A noise-resilient defect classification framework using memory-augmented auto-encoders with trust regions [[2021]](http://arxiv.org/pdf/2012.14629)
 + (yoon2021self)Self-Supervise, Refine, Repeat: Improving Unsupervised Anomaly Detection [[2021]](https://arxiv.org/pdf/2106.06115.pdf)
 + (cordier2022data)Data refinement for fully unsupervised visual inspection using pre-trained networks [[2022]](https://arxiv.org/pdf/2202.12759.pdf)
 + (qiu2022latent)Latent Outlier Exposure for Anomaly Detection with Contaminated Data [[2022]](https://arxiv.org/pdf/2202.08088.pdf)
 + (chen2022deep)Deep one-class classification via interpolated gaussian descriptor [[2022]](https://arxiv.org/pdf/2101.10043.pdf)
 + (roth2022towards)Towards total recall in industrial anomaly detection [[2022]](http://arxiv.org/pdf/2106.08265)
 + (xisoftpatch)SoftPatch: Unsupervised Anomaly Detection with Noisy Data [[2020]](http://arxiv.org/pdf/2009.09443)
 + (bergmann2019mvtec)MVTec AD: A comprehensive real-world dataset for unsupervised anomaly detection [[2019]](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/mvtec_ad.pdf)

## 3.3 Anomaly Synthetic
 + (li2021cutpaste)Cutpaste: Self-supervised learning for anomaly detection and localization [[2021]](http://arxiv.org/pdf/2104.04015)
 + (yang2022memseg)MemSeg: A semi-supervised method for image surface defect detection using differences and commonalities [[2022]](https://arxiv.org/pdf/2205.00908.pdf)
 + (liu2019multistage)Multistage GAN for fabric defect detection [[2019]](https://pubmed.ncbi.nlm.nih.gov/31870985/)
 + (rippel2020gan)Gan-based defect synthesis for anomaly detection in fabrics [[2020]](https://www.lfb.rwth-aachen.de/bibtexupload/pdf/RIP20c.pdf)
 + (zhu2017unpaired)Unpaired image-to-image translation using cycle-consistent adversarial networks [[2017]](https://namanuiuc.github.io/assets/projects/CGAN/report.pdf)
 + (niu2020defect)Defect image sample generation with GAN for improving defect recognition [[2020]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9000806)
 + (wei2020defective)Defective samples simulation through neural style transfer for automatic surface defect segment [[2020]](http://arxiv.org/pdf/1910.03334)
 + (wei2020simulation)A simulation-based few samples learning method for surface defect segmentation [[2020]](https://www.sciencedirect.com/science/article/pii/S0925231220310791/pdfft?md5=f3f72bc8687c8f9968d4a2a1bd3ea17e&pid=1-s2.0-S0925231220310791-main.pdf)
 + (jain2020synthetic)Synthetic data augmentation for surface defect detection and classification using deep learning [[2020]](https://link.springer.com/article/10.1007/s10845-020-01710-x)
 + (wang2021defect)Defect Transfer GAN: Diverse Defect Synthesis for Data Augmentation [[2021]](https://openreview.net/pdf?id=2hMEdc35xZ6)
 + (heusel2017gans)Gans trained by a two time-scale update rule converge to a local nash equilibrium [[2017]](https://arxiv.org/pdf/1706.08500.pdf)
 + (binkowski2018demystifying)Demystifying MMD GANs [[2018]](https://arxiv.org/pdf/1801.01401.pdf)
 + (zhang2021defect)Defect-GAN: High-fidelity defect synthesis for automated defect inspectio [[2021]](https://dr.ntu.edu.sg/bitstream/10356/146285/2/WACV_2021_Defect_GAN__Camera_Ready_.pdf)
 
## 3.4 3D AD
 + ({bergmann2022beyond)Beyond Dents and Scratches: Logical Constraints in Unsupervised Anomaly Detection and Localization[[2022]](https://link.springer.com/content/pdf/10.1007/s11263-022-01578-9.pdf)
 + (bergmann2022anomaly)Anomaly detection in 3d point clouds using deep geometric descriptors [[2022]](https://arxiv.org/pdf/2202.11660.pdf)
 + (horwitz2022back)Back to the feature: classical 3d features are (almost) all you need for 3D anomaly detection [[2022]](https://arxiv.org/pdf/2203.05550.pdf)
 + (rusu2009fast)Fast point feature histograms (FPFH) for 3D registration [[2009]](http://www.cvl.iis.u-tokyo.ac.jp/class2016/2016w/papers/6.3DdataProcessing/Rusu_FPFH_ICRA2009.pdf)
 + (roth2022towards)Towards total recall in industrial anomaly detection [[2022]](http://arxiv.org/pdf/2106.08265)
 + (reiss2022anomaly)Anomaly Detection Requires Better Representations [[2022]](https://arxiv.org/pdf/2210.10773.pdf)
 + (rudolph2022asymmetric)Asymmetric Student-Teacher Networks for Industrial Anomaly Detection [[2022]](https://arxiv.org/pdf/2210.07829.pdf)
 
## 3.5 Continual AD
 + (li2022towards)Towards Continual Adaptation in Industrial Anomaly Detection [[2022]](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8682702)

## 3.6 Uniform AD
 + (you2022unified)A Unified Model for Multi-class Anomaly Detection [[2022]](https://arxiv.org/pdf/2206.03687.pdf)
## 3.7 Logical AD
 + (bergmann2022beyond)Beyond Dents and Scratches: Logical Constraints in Unsupervised Anomaly Detection and Localization [[2022]](https://link.springer.com/content/pdf/10.1007/s11263-022-01578-9.pdf)
 
# 4 Dataset
 + (song2013noise)A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects [[2013]](https://www.sciencedirect.com/science/article/pii/S0169433213016437/pdfft?md5=478bf7f07bbf551a5d991048f9bc16e4&pid=1-s2.0-S0169433213016437-main.pdf)
 + (yang2021deep)Deep learning based steel pipe weld defect detection [[2021]](https://www.tandfonline.com/doi/pdf/10.1080/08839514.2021.1975391?needAccess=true)
 + (SDD2019)Severstal: Steel Defect Detection [[2019]]()
 + (carrera2016defect)Defect detection in SEM images of nanofibrous materials [[2016]](https://re.public.polimi.it/bitstream/11311/1024586/1/anomaly_detection_sem.pdf)
 + (mery2015gdxray)GDXray: The database of X-ray images for nondestructive testing [[2015]](http://dmery.sitios.ing.uc.cl/Prints/ISI-Journals/2015-JNDE-GDXray.pdf)
 + (tang2019online)Online PCB defect detector on a new PCB defect dataset [[2019]](https://arxiv.org/pdf/1902.06197.pdf)
 + (tsang2016fabric)Fabric inspection based on the Elo rating method [[2016]](http://hub.hku.hk/bitstream/10722/229176/1/content.pdf)
 + (tabernik2020segmentation)Segmentation-based deep-learning approach for surface-defect detection [[2020]](http://arxiv.org/pdf/1903.08536)
 + (bovzivc2021mixed)Mixed supervision for surface-defect detection: From weakly to fully supervised learning [[2021]](https://arxiv.org/pdf/2104.06064.pdf)
 + (gan2017hierarchical)A hierarchical extractor-based visual rail surface inspection system [[2017]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8063875)
 + (bonfiglioli2022eyecandies)The Eyecandies Dataset for Unsupervised Multimodal Anomaly Detection and Localization [[2022]](https://arxiv.org/pdf/2210.04570.pdf)
 + (bergmann2019mvtec)MVTec AD: A comprehensive real-world dataset for unsupervised anomaly detection [[2019]](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/mvtec_ad.pdf)
 + (bergmann2021mvtec)The mvtec 3d-ad dataset for unsupervised 3d anomaly detection and localization [[2021]](https://arxiv.org/pdf/2112.09045.pdf)
 + (bergmann2022beyond)Beyond Dents and Scratches: Logical Constraints in Unsupervised Anomaly Detection and Localization [[2022]](https://link.springer.com/content/pdf/10.1007/s11263-022-01578-9.pdf)
 + (jezek2021deep)Deep learning-based defect detection of metal parts: evaluating current methods in complex conditions [[2021]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9631567)
 + (mishra2021vt)VT-ADL: A vision transformer network for image anomaly detection and localization [[2021]](http://arxiv.org/pdf/2104.10036)
 + (zou2022spot)SPot-the-Difference Self-supervised Pre-training for Anomaly Detection and Segmentation [[2022]](https://arxiv.org/pdf/2207.14315.pdf)
 + (huang2020surface)Surface defect saliency of magnetic tile [[2020]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8560423)
 + (DAGMGNSS2077)DAGM dataset [[2000]]()
 
