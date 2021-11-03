# [![website](https://img.shields.io/static/v1?label=&message=wav2mov&color=blue&style=for-the-badge)](https://wav2mov.vercel.app)

##  Speech To Facial Animation Using GANs


[![python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) [![pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/) [![GANs](https://img.shields.io/badge/GANs-4BB749?style=for-the-badge&logo=&logoColor=white)](#1)

This repo contains the pytorch implementation of achieving facial animation from given face image and speech input using Generative Adversarial Nets (See [References](#1)).


## Results

Some of the generated videos are found [here](https://wav2mov-examples.vercel.app/examples).

## Implementation
### GAN setup
![gan_setup](/wav2mov-docs/gan_setup.PNG)
![generator_architecture](/wav2mov-docs/gen_arch.PNG)
## References

<a id="1" href="https://arxiv.org/abs/1406.2661">[1] Generative Adversarial Nets</a>
```bibtex
@article{goodfellow2014generative,
  title={Generative adversarial networks},
  author={Goodfellow, Ian J and Pouget-Abadie, Jean and Mirza, Mehdi and Xu, Bing and Warde-Farley, David and Ozair, Sherjil and Courville, Aaron and Bengio, Yoshua},
  journal={arXiv preprint arXiv:1406.2661},
  year={2014}
}
```

<a id="2" href="http://spandh.dcs.shef.ac.uk/avlombard/">[2] The Audio-Visual Lombard Grid Speech Corpus</a>

[![Github stars](https://img.shields.io/badge/Dataset-LombardGrid-<COLOR>.svg)](http://spandh.dcs.shef.ac.uk/avlombard/)

```bibtex
@article{Alghamdi_2018,
	doi = {10.1121/1.5042758},
	url = {https://doi.org/10.1121%2F1.5042758},
	year = 2018,
	month = {jun},
	publisher = {Acoustical Society of America ({ASA})},
	volume = {143},
	number = {6},
	pages = {EL523--EL529},
	author = {Najwa Alghamdi and Steve Maddock and Ricard Marxer and Jon Barker and Guy J. Brown},
	title = {A corpus of audio-visual Lombard speech with frontal and profile views},
	journal = {The Journal of the Acoustical Society of America}
}
```

<a id="#3" href="https://link.springer.com/article/10.1007/s11263-019-01251-8">[3] Realistic Facial Animation using GANs</a>

[![Github stars](https://img.shields.io/badge/Github-sda-<COLOR>.svg)](https://github.com/DinoMan/speech-driven-animation)

```bibtex
@article{Vougioukas_2019,
	doi = {10.1007/s11263-019-01251-8},
	url = {https://doi.org/10.1007%2Fs11263-019-01251-8},
	year = 2019,
	month = {oct},
	publisher = {Springer Science and Business Media {LLC}},
	volume = {128},
	number = {5},
	pages = {1398--1413},
	author = {Konstantinos Vougioukas and Stavros Petridis and Maja Pantic},
	title = {Realistic Speech-Driven Facial Animation with {GANs}},
	journal = {International Journal of Computer Vision}
}
```

<a id="#4" href="https://arxiv.org/abs/1805.09313">[4] End to End Facial Animation using Temporal GANs</a>
```bibtex
@article{vougioukas2018end,
  title={End-to-end speech-driven facial animation with temporal gans},
  author={Vougioukas, Konstantinos and Petridis, Stavros and Pantic, Maja},
  journal={arXiv preprint arXiv:1805.09313},
  year={2018}
}
```

<a id="#5" href="https://arxiv.org/abs/2008.10010">[5] A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild</a>


[![Github stars](https://img.shields.io/badge/Github-wav2Lip-<COLOR>.svg)](https://github.com/Rudrabha/Wav2Lip)
```bibtex
@inproceedings{10.1145/3394171.3413532,
author = {Prajwal, K R and Mukhopadhyay, Rudrabha and Namboodiri, Vinay P. and Jawahar, C.V.},
title = {A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild},
year = {2020},
isbn = {9781450379885},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3394171.3413532},
doi = {10.1145/3394171.3413532},
booktitle = {Proceedings of the 28th ACM International Conference on Multimedia},
pages = {484–492},
numpages = {9},
keywords = {lip sync, talking face generation, video generation},
location = {Seattle, WA, USA},
series = {MM '20}
}
```
