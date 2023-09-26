# SleepWalker
Dream booth implementation for Text To Motion models to fine tune to YOUR motion gate.


## Acknowledgements
The inspiration for this work is the Text to Motion diffusion model from Eric Guo as well as my previous expierence at X:
https://github.com/EricGuo5513/text-to-motion/tree/main


This work wouldnt be possible without the help of the following:

Google Researchs' Dreambooth paper here
@article{ruiz2022dreambooth,
  title={DreamBooth: Fine Tuning Text-to-image Diffusion Models for Subject-Driven Generation},
  author={Ruiz, Nataniel and Li, Yuanzhen and Jampani, Varun and Pritch, Yael and Rubinstein, Michael and Aberman, Kfir},
  booktitle={arXiv preprint arxiv:2208.12242},
  year={2022}
}

HuggingFaces implentation of Dreambooth:
https://huggingface.co/docs/diffusers/training/dreambooth

Keras Implementation of Dreambooth:
https://keras.io/examples/generative/dreambooth/

As well as Xavier Zhao's implementation of DreamBooth:
https://github.com/XavierXiao/Dreambooth-Stable-Diffusion/tree/main

## File Structure and Setup

To set up the project, follow these steps:

1. **Download Checkpoints**: You can download the checkpoints from the following link: HumanML3D[Checkpoints](https://drive.google.com/file/d/1DSaKqWX2HlwBtVH5l7DdW96jeYUIXsOP/view?pli=1) KITML[Checkpoints](https://drive.google.com/file/d/1DSaKqWX2HlwBtVH5l7DdW96jeYUIXsOP/view?pli=1) . After downloading, place the checkpoints in the `checkpoints/` directory.

2. **Add Datasets**: Place your datasets in the `datasets/` directory. Make sure to follow the same structure as the existing datasets for compatibility with the code. Here is a link to the orignial repo to download the HumanML3D dataset and KitML https://github.com/EricGuo5513/text-to-motion/tree/main

3. **Download GloVe**: GloVe embeddings are used for word vectorization. You can download them from the following link: [GloVe](http://nlp.stanford.edu/data/glove.6B.zip). After downloading, extract the contents and place them in the `glove/` directory.

Your directory structure should look like this:
.
├── checkpoints
│   └── Comp_v6_KLD01
│       ├── meta
│       │   └── Compv6_architecture.py
│       └── opt.txt
├── data
│   └── dataset.py
├── main.py
├── models.py
├── README.md
├── utils
│   ├── build.py
│   ├── opt.py
│   └── word_vectorizer.py
└───dataset
│   └── HumanML3D
│       ├── HumanML3D