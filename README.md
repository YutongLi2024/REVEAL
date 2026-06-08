# REVEAL

REVEAL is a plug-and-play framework for enhancing the effectiveness of visual features in Multimodal Sequential Recommendation (MSR).

## Settings

```
python = 3.10
pytorch = 2.9.0
cuda = 13.0
```

## Structure

```
├HM4SR/
├──...
│
├M3SRec/
├──...
│
├MISSRec/
├──...
│
├REVEAL/
├── AVL/
│   ├── M3SRec.py        # M3SRec backbone with AVL support
│   └── trainers.py     # Trainer with Adaptive Visual Learning
│
├── PVE/
│   ├── prompt_templates/
│   ├── critic.py
│   └── prompt_optimizer.py
│
├── README.md

```

## DataProcessing

REVEAL follows the original data preprocessing pipeline of each MSR backbone.
Please refer to the corresponding model directory (e.g., M3SRec, HM4SR, or MISSRec) for dataset preparation and preprocessing scripts. For example, to prepare data for M3SRec:

```
cd M3SRec
python DataProcessing.py
```

```
cd REVEAL
cd PVE
```

## Training with REVEAL

REVEAL is designed as a plug-and-play framework and is applied on top of existing MSR backbones.
To run REVEAL, please enter the directory of the target backbone and enable the corresponding modules.

```
cd M3SRec
python main.py
```

Personalized Visual Extraction (PVE) is performed to refine visual representations before training.

Adaptive Visual Learning (AVL) is enabled during training to dynamically calibrate visual gradients.

The original training and inference pipelines of the backbone remain unchanged.

Other MSR backbones (e.g., HM4SR, MISSRec) can be used in the same manner by switching to the corresponding directory.

## Acknowledgements

Our code is based on the implementation of [RecBole](https://github.com/RUCAIBox/RecBole).
