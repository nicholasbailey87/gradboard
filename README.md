# gradboard
![snowboarder](snowboarder.png "Image of a snowboarder")

Easily snowboard down gnarly loss gradients

## Getting started

You can install gradboard with

```
pip install gradboard
```

PyTorch is a peer dependency of `gradboard`, which means
  * You will need to make sure you have PyTorch installed in order to use `gradboard`
  * PyTorch will **not** be installed automatically when you install `gradboard`

We take this approach because PyTorch versioning is environment-specific and
    we don't know where you will want to use `gradboard`. If we automatically install
    PyTorch for you, there's a good chance we would get it wrong!

Therefore, please also make sure you install PyTorch.

## Usage examples

### Decent model training outcomes without tuning hyperparameters

`gradboard` includes

  * An implementation of AdamS as proposed in Xie et al. (2023) "On the Overlooked
        Pitfalls of Weight Decay and How to Mitigate Them: A Gradient-Norm
        Perspective" (https://openreview.net/pdf?id=vnGcubtzR1), which in practice
        makes model training more robust to the weight decay setting.
  * Utilities for implementing popular learning rate schedules
  * An implementation of an automatic max/min learning rate finder based on Smith
        (2017) "Cyclical Learning Rates for Training Neural Networks"
        (https://arxiv.org/abs/1506.01186)
  * Sensible defaults

In practice this means that you can train a neural network and get decent performance
    right out of the box, just by using the `PASS` (point-and-shoot scheduler), even
    for unfamiliar architectures or problem domains.



