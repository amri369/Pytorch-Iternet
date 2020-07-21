# Pytorch-Iternet

PyTorch implementation of IterNet, based on paper IterNet: Retinal Image Segmentation Utilizing Structural Redundancy
in Vessel Networks [(Li et al., 2019)](https://arxiv.org/abs/1912.05763) and accompanying [code](https://github.com/conscienceli/IterNet).

# Training parameters

```bash
python main.py --param param_value
```

The following hyperparameters can also be provided. Smallest model from paper is
shown for comparison.

Argument          | Default
---               | ---
`--gpus`          | 4,5,6
`--size.`         | 592
`--crop_size`     | 128
`--image_dir`     | ''
`--mask_dir`      | ''
`--train.csv`     | ''
`--val.csv`       | ''
`--lr`            | 0.0001
`--epochs`        | 2
`--batch_size`    | 32
`--model_dir`     | ''
`--seed`          | 2020123
