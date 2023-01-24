# HiFi-VC: High Quality ASR-Based Voice Conversion

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2203.16937)

### Anton Kashkin, Ivan Karpukhin, Svyatoslav Shishkin


We provide our pretrained model and notebook for inference in this repository.

**Abstract :**
The goal of voice conversion (VC) is to convert input voice to match the target speaker's voice while keeping text and prosody intact. VC is usually used in entertainment and speaking-aid systems, as well as applied for speech data generation and augmentation. The development of any-to-any VC systems, which are capable of generating voices unseen during model training, is of particular interest to both researchers and the industry. Despite recent progress, any-to-any conversion quality is still inferior to natural speech.
In this work, we propose a new any-to-any voice conversion pipeline. Our approach uses automated speech recognition (ASR) features, pitch tracking, and a state-of-the-art waveform prediction model. According to multiple subjective and objective evaluations, our method outperforms modern baselines in terms of voice quality, similarity and consistency.

### Paper demo: [samples](https://paint-kitten-d96.notion.site/2fbe30b894a64f7fa8bccb96f8d09540)

### Pre-trained model: [google drive](https://drive.google.com/file/d/1oFwMeuQtwaBEyOFkyG7c7LfBQiRe3RdW/view?usp=share_link). put the model to the root of the repository.

## Setup environment:

1) Using docker (recommended):

    Build docker image:
    ```sh
    docker build . -t hifi_vc
    ```

    run docker with exact one gpu!
    ```sh
    docker run --gpus '"device=0"' -it --net=host hifi_vc
    ```

2. Using Pip (use `torch>=1.13`):

    ```sh
    pip install -r requirements
    ```

## Inference:

1. Download the model.
2. Build a docker.
3. Inference with [`inference.ipynb`](inference.ipynb)

## Acknowledgements:

The `f0_utils.py` in modified from [PPG-VC]( https://github.com/liusongxiang/ppg-vc)

## License

Feel free to use our library in your commercial and private applications.

hifi_vc is covered by [Apache 2.0](/LICENSE). 
Read more about this license [here](https://choosealicense.com/licenses/apache-2.0/)