# Observations 🔍

***

+ __sync discriminator loss is very small (of the order of e-9)__

    + solved by using removing batch normalization and using leaky version of relu with slope of 0.2 instead of vanilla relu.

+ __changes in losses is very small__

## lip variations
    + it requires longer times to get variations .
        + size of dataset plays huge role
        (60 videos with 2 actores took nearly 200 epochs to show mild variations between frames)
    + train sync discriminator only on real frames initially both on synchronized and unsynchronized audio and video frames.
