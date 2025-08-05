<div align="center">
<h1>
  MedITok: A Unified Tokenizer for Medical Image Synthesis and Interpretation
</h1>
</div>

<p align="center">
📝 <a href="https://arxiv.org/abs/2505.19225" target="_blank">Paper</a> • 🤗 <a href="https://huggingface.co/massaki75/meditok/tree/main" target="_blank">Hugging Face</a> • 🧩 <a href="https://github.com/Masaaki-75/meditok" target="_blank">Github</a>
</p>

This is the official repository for MedITok, a unified visual tokenizer tailored for medical image. MedITok encodes both low-level details and high-level semantics into a unified token space, and supports building strong generative models for a wide range of tasks including medical image synthesis and interpretation. 


## 📌 Overview
![](./assets/arch.png)


## 🚧 Project Status
- [x] Release [preprint](https://arxiv.org/abs/2505.19225).
- [x] Release the initial [weights](https://huggingface.co/massaki75/meditok/tree/main).
- [x] Release evaluation code.
- [x] Release training code.


## 🎬 Demo
1. Put the downloaded checkpoint file `meditok_simple_v1.pth` in `weights/meditok` folder. 
2. Create a virtual environment with core libraries listed in `requirements.txt`. 
3. Open `demo.ipynb` and click `Run All` to run the reconstruction demo. Feel free to change the images you would like to play with. 
4. Run `python demo.py` to save the reconstruction results. 

## 🔥 Training
Before training / fine-tuning the MedITok model, we need to:
1. Download pretrained weights ([ViTamin](https://huggingface.co/jienengchen/ViTamin-B), [BiomedClip](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/tree/main), [BiomedBERT](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract/tree/main), etc.) and fill the local paths in `./local_openclip/constants.py`
2. Download the [models](https://huggingface.co/FoundationVision/unitok_external) used for loss calculation, create a folder named `./external` and put the models under it.
3. Write the metadata as a `.csv` file with columns of `"identifier"` (relative or absolute path of each image), `"caption"` (the paired caption), and `"modality"` (imaging modality of the image).
  - Note that, we save each CT slice as an `int16` PNG file to preserve the HU values, which allows for CT windowing data augmentation. Thus images tagged with `"modality"=="ct"` would undergo specific preprocessing (see the `ReadMedicalImage` class in `./datasets/transforms.py` for detail).
4. Configure the variables in the training scripts (`./scripts/train_stage1.sh` and `./scripts/train_stage2.sh`). To figure out what each variable represent, please see the `Args` class in `./utilities/config.py`.

Once we have everything prepared, we can run the scripts in `./scripts` to launch the training. If you catch any bugs, feel free to open an issue/PR!


## 🙏 Acknowledgment
This project is built upon and inspired by several excellent prior works:
- [UniTok](https://github.com/FoundationVision/UniTok)
- [LlamaGen](https://github.com/FoundationVision/LlamaGen)
- [LLaVA-Med](https://github.com/microsoft/LLaVA-Med)
- ...

The model also benefits from many publicly available medical image datasets. We kindly refer readers to our preprint for details.

We sincerely thank the communities behind these works for making the resources available and inspiring further research in the field. 


## 🚀 Notes
If you build something exciting or encounter any issues when using our model, please feel free to open an issue, submit a pull request, or contact us with feedback. Your contributions and insights are highly valued!


## 📖 Citation
If you find MedITok useful for your research and applications, please kindly cite our work:
```
@article{ma2025meditok,
  title={{MedITok}: A Unified Tokenizer for Medical Image Synthesis and Interpretation},
  author={Ma, Chenglong and Ji, Yuanfeng and Ye, Jin and Li, Zilong and Wang, Chenhui and Ning, Junzhi and Li, Wei and Liu, Lihao and Guo, Qiushan and Li, Tianbin and He, Junjun and Shan, Hongming},
  journal={arXiv preprint arXiv:2505.19225},
  year={2025}
}
```


