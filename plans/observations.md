# Observations 🔍

***

+ __sync discriminator loss is very small (of the order of e-9)__

    + solved by using removing batch normalization and using leaky version of relu with slope of 0.2 instead of vanilla relu.

+ __changes in losses is very small__

