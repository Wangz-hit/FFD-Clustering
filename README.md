# FFD-Clustering

**Unsupervised anomaly detection for aero-engine monitoring data based on fuzzy fusion of variables and discriminative mapping of features.**

[English](README.md) | [简体中文](README.zh-CN.md)

FFD-Clustering is an implementation of the framework presented in:

Wang, Z., Fu, X., Zhao, M., Xia, X., & Zhong, S. (2025). [*FFD-Clustering: An unsupervised anomaly detection method for aero-engines based on fuzzy fusion of variables and discriminative mapping of features*](https://doi.org/10.1016/j.cja.2024.103346). *Chinese Journal of Aeronautics, 38*(5), 103346.

## Overall Framework

<p align="center">
  <img src="figures/overall_framework.png" alt="Overall framework of FFD-Clustering" width="850">
</p>

*Fig. 1. Overall framework of FFD-Clustering (from the cited article).*

## Method

FFD-Clustering consists of four stages:

1. **Fuzzy Grouping (FG).** K-means-initialized Fuzzy C-Means (FCM) partitions associated sensor variables into overlapping groups according to fuzzy memberships. Partition Coefficient (PC) and Classification Entropy (CE) support group-number selection.
2. **Fusion Encoding Network (FEN).** A query encoder, key encoder and query decoder learn representative features for each variable group through reconstruction and multi-layer contrastive learning.
3. **Discriminative Mapping Network (DMN).** Reconstruction-error re-clustering guides the mapping of group features into a unified feature space.
4. **Anomaly Detection.** K-means operates on the mapped features. The anomaly score is the Euclidean distance to the nearest centroid:

   $$
   s(\mathbf z_i) = \min_{k} \|\mathbf z_i - \mathbf c_k\|_2.
   $$

   A quantile of training anomaly scores determines the binary detection threshold. The FEN decoders provide reconstruction-based rankings of associated variable groups for anomaly localization.

### Fusion Encoding Network (FEN)

Each fuzzy group is processed by a Fusion Encoding Network with a query encoder, key encoder, and query decoder. The query network is trained with reconstruction and contrastive objectives; the key encoder provides representations for contrastive learning.

<p align="center">
  <img src="figures/fig5_fusion_encoding_network.png" alt="Fusion Encoding Network architecture" width="760">
</p>

*Fig. 5. Multi-layer contrastive denoising-based feature fusion encoding network (from the cited article).*

### Discriminative Mapping Network (DMN)

The representative group features are mapped into a unified feature space. The DMN uses overall reconstruction, reconstruction-error-based re-clustering, and a center-distance objective during training.

<p align="center">
  <img src="figures/fig8_discriminative_mapping_network.png" alt="Discriminative Mapping Network training procedure" width="850">
</p>

*Fig. 8. Four steps of discriminative mapping with reconstruction-difference re-clustering (from the cited article).*

The cluster centers used during DMN training are updated progressively to stabilize the feature mapping process.

<p align="center">
  <img src="figures/fig9_dynamic_cluster_center_update.png" alt="Dynamic cluster center update during DMN training" width="780">
</p>

*Fig. 9. Dynamic cluster-center updating during DMN training (from the cited article).*

## Installation

Python 3.10 or later is required.

```bash
python -m pip install -r requirements.txt
```

## Usage

Provide training and test CSV files with the same numeric sensor columns in the same order. The input CSV files should not contain label or timestamp columns.

```bash
python run.py --train-csv train.csv --test-csv test.csv --groups 6 --clusters 4 --contamination 0.05
```

Use `--groups` to set the number of fuzzy variable groups and `--clusters` to set the number of K-means clusters. With `--scan-groups`, the program saves PC and CE values for candidate group counts. `--contamination` specifies the assumed anomaly fraction used to set the threshold from training scores.

Use `--output-dir` to choose the output directory (default: `outputs/run/`). The program saves the fitted model, fuzzy memberships, mapped features, anomaly scores, binary predictions, and variable-group localization results.

## Citation

```bibtex
@article{wang2025ffd,
  title={FFD-Clustering: An unsupervised anomaly detection method for aero-engines based on fuzzy fusion of variables and discriminative mapping of features},
  author={Wang, Zhe and Fu, Xuyun and Zhao, Minghang and Xia, Xiangzhao and Zhong, Shisheng},
  journal={Chinese Journal of Aeronautics},
  volume={38},
  number={5},
  pages={103346},
  year={2025},
  doi={10.1016/j.cja.2024.103346}
}
```
