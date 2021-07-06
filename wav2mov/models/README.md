# Architectures

## Generator

* follows UNET architecture
* input : audio_frames (B,F,Sw) and ref_frames (B,F,C,H,W)
* out is IMAGE_SIZE*IMAGE_SIZE

## identity Descriminator

* input is real/fake frame and still face image(ref face image)
* helps to retain identity of face of given reference image and to produce realistic images.

## sequence Descriminator

* input is  consecutive frames
* helps to produce cohesive video frames.

## sync Descriminator

* input is audio frame(666 points*5) and corresponding video frames.
* helps to synchronize produced video frames with that of audio.

