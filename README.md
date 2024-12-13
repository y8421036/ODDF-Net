

# ODDF-Net: Multi-object segmentation in 3D retinal OCTA using optical density and disease features

We propose a 3D-to-2D multi-object segmentation network based on Optical Density and disease features, which can jointly extract 2D capillaries, arteries, veins, and FAZ from 3D OCTA.

This work has been published at "Knowledge-Based Systems".

DOI: [10.1016/j.knosys.2024.112704](https://www.sciencedirect.com/science/article/abs/pii/S0950705124013388?via%3Dihub)

## Requirements
python>=3.8.0

torch>=1.6.0

## Dataset Download Link
OCTA-500: [https://ieee-dataport.org/open-access/octa-500](https://ieee-dataport.org/open-access/octa-500)

## Usage
* First, Data preprocessing is performed and OD modality is generated.
* Then, train the model without classification header using 'train_seg.py' and save the model parameters.
* Next, train the full model containing classification header using 'train_cla.py' and save the model parameters.
* Finally, use 'test.py' to run these two models on the test set to obtain quantitative and qualitative results.
* **Note: The hyperparameters related to the training and testing phases need to be set in 'base_options.py', 'train_options.py', and 'test_options.py'.**

## Citation
If this code is helpful for your study, please cite:
```
@article{yang2024oddf,
  title={ODDF-Net: Multi-object segmentation in 3D retinal OCTA using optical density and disease features},
  author={Yang, Chaozhi and Fan, Jiayue and Bai, Yun and Li, Yachuan and Xiao, Qian and Li, Zongmin and Li, Hongyi and Li, Hua},
  journal={Knowledge-Based Systems},
  pages={112704},
  year={2024},
  publisher={Elsevier}
}
```
