# 3D CAD Part Classification with PointNet++

An end-to-end geometric deep learning pipeline that converts 3D CAD surface meshes into unit-sphere-normalized point clouds and classifies 3D object geometries using **PointNet++ (Single-Scale Grouping)** in PyTorch.

---

## Overview & Motivation
Traditional 2D convolutional networks cannot directly process unstructured 3D CAD surfaces without costly voxelization. This project implements a direct mesh-to-point-cloud workflow suitable for automated CAD part recognition and digital engineering pipelines:
1. **Surface Point Sampling (`trimesh`):** Uniformly samples `1,024` surface coordinates `(x, y, z)` per 3D CAD mesh across 10 object categories (`3,991` training models / `908` unseen test models from ModelNet10).
2. **Geometric Normalization & Augmentation:** Centers each point cloud at the origin `(0, 0, 0)`, normalizes coordinates to a unit sphere, and applies on-the-fly Z-axis rotational augmentation and Gaussian jitter (`σ = 0.01`) during training.
3. **Hierarchical Feature Learning (`PointNet++`):** Uses Set Abstraction (SA) layers from `pointnet2_cls_ssg` (`normal_channel=False`) to capture both local surface neighborhoods and global part geometry.

---

## Repository Structure
```text
3d-cad-part-classification/
├── 3D_CAD_PointNet2_Classifier.ipynb   # End-to-end Google Colab pipeline (data, training, eval)
├── best_pointnet2_cad.pth              # Best saved PointNet++ model checkpoint (Epoch 14)
├── training_curves.png                 # Training loss & train/test accuracy progression
├── confusion_matrix.png                # Per-class classification confusion matrix
├── cad_predictions_grid.png            # 3D point cloud inference visualizations
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
