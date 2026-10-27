# DeepRM
Deep learning for RNA Modification

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-red?logo=github)](https://github.com/vadanamu/DeepRM)
[![CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-blue)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
![GitHub Repo stars](https://img.shields.io/github/stars/vadanamu/DeepRM?style=social)
![GitHub last commit](https://img.shields.io/github/last-commit/vadanamu/DeepRM)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/vadanamu/DeepRM)
![GitHub contributors](https://img.shields.io/github/contributors/vadanamu/DeepRM)
![GitHub language count](https://img.shields.io/github/languages/count/vadanamu/DeepRM)
![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17636697.svg)

```{toctree}
:maxdepth: 1
installation
quickstart
usage
advanced-usage
advanced-installation
troubleshooting
cli/index
api/index
changelog
license
```

## ✨ Introduction
DeepRM is a deep learning-based framework for RNA modification detection using Nanopore direct RNA sequencing.
This repository contains the source code for training and running DeepRM.

(whats-new-v12)=
## 🆕 What's new in v1.2
DeepRM v1.2 uses substantially less memory, runs faster and detects RNA modifications more accurately.

**1. Substantially lower memory usage and higher speed**
* The preprocessing uses 8 to 13 times less peak memory than DeepRM v1.1's.
* With input on a local SSD, the preprocessing runs 2 to 2.5 times as fast as DeepRM v1.1's. Run on Dorado's output
  while Dorado basecalls, it finishes when basecalling does, so it adds no time to the run.
* Inference is 1.9 to 3.0 times as fast as DeepRM v1.1 on A100, H100 and B200 GPUs.

**2. Improved RNA modification detection accuracy**
* Higher m6A recall, most of all for clustered m6A (the nearest other m6A 1–4 nt away): 95.7 %, against 85.8 % with
  DeepRM v1.1 and 78.9 % with Dorado (v6.0.0 SUP), at the same false positive rate.
* Oxford Nanopore Technologies' all-5-mer synthetic RNA dataset: AU-PR 0.933, against 0.883 with DeepRM v1.1 and 0.923
  with Dorado.
* HEK293T benchmark of the DeepRM paper: AU-PR 0.964 at DRACH and 0.716 at non-DRACH sites, against 0.961 and 0.704 with
  DeepRM v1.1, and 0.959 and 0.161 with Dorado.

Benchmark details and the full list of changes: {doc}`changelog`.

## 🎯 Key Features
* **High accuracy**: Achieves state-of-the-art accuracy in RNA modification detection and stoichiometry measurement.
* **Single-molecule resolution**: Provides single-molecule level predictions for RNA modifications.
* **End-to-end pipeline**: Easy-to-use pipeline from raw reads to molecule-level and site-level predictions.
* **Fast**: Compiled inference and a fast C++ preprocessing.
* **Customizable**: Supports training of custom models.


## 📝 Citation
If you use DeepRM in your research, please cite the following paper:
```{code-block} text
:class: nohighlight
@article{kang2026deeprm,
  title={Comprehensive discovery of m6A sites in the human transcriptome at single-molecule resolution},
  author={Kang, Gihyeon and Hwang, Hyeonseo and Jeon, Hyeonseong and Choi, Heejin and Chang, Hee Ryung and Yeo, Nagyeong and Park, Junehee and Son, Narae and Jeon, Eunkyeong and Lim, Jungmin and Yun, Jaeung and Choi, Wook and Jo, Jae-Yoon and Kim, Jong-Seo and Park, Sangho and Kim, Yoon Ki and Baek, Daehyun},
  journal={Nature Communications},
  year={2026},
  volume={17},
  pages={664},
  publisher={Springer Nature},
  doi={10.1038/s41467-025-67417-w}
}
```
The article is fully open access and available at <https://doi.org/10.1038/s41467-025-67417-w>.


## 🏛 Contributors
This repository is developed and maintained by the following organization:
* **Laboratory of Computational Biology, School of Biological Sciences, Seoul National University**
    * Principal Investigator: Prof. Daehyun Baek
* **Genome4me, Inc., Seoul, Republic of Korea**


## 🏛 Acknowledgements
This study was supported by the National Research Foundation of Korea (NRF) funded by the Ministry of Science and ICT, Republic of Korea (MSIT) (RS-2019-NR037866, RS-2020-NR049252, RS-2020-NR049538, and RS-2022-NR067483), by a grant of Korean ARPA-H Project through the Korea Health Industry Development Institute (KHIDI), funded by the Ministry of Health & Welfare, Republic of Korea (RS-2025-25422732), by Artificial Intelligence Industrial Convergence Cluster Development Project funded by MSIT and Gwangju Metropolitan City, by National IT Industry Promotion Agency (NIPA) funded by MSIT, and by Korea Research Environment Open Network (KREONET) managed and operated by Korea Institute of Science and Technology Information (KISTI).
