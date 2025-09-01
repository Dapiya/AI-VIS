# AI-VIS

AI-VIS is a Conditional GAN(CGAN) based model that simulates visible imagery from multiple IR channels of geostationary weather satellites at night.

The model is trained on Himawari-8/9 Full Disk and Target Area data. The model has been tested on other modern satellites including GOES-R series and GK-2A. Support is expected in the future.

AI-VIS 1.0 is the model presented in the paper. Since then, we have continued to improve the model, primarily by introducing LPIPS loss, as well as increasing the number of filters of the U-Net. AI-VIS 1.0 is available to everyone on HuggingFace. If you're interested in our most advanced model, please fill out the request form(see table below) to get access.

Earlier iterations of AI-VIS is trained with much less data, and a different set of inputs. It takes a significant amount of extra code to support it, with little real-world use at this stage, this repo does not support versions earlier than 1.0.

[2025/06/09] We have released a set of test data for anyone who wants to take a look but without the hassle of configuring everything: [Google Drive](https://drive.google.com/drive/folders/1AHVWuQ9z1P_8nujVWygW4YDNl-6iu3Zq?usp=sharing) [Dapiya](https://archive.dapiya.top/Custom/aivis_test_h9/)


**License: Apache 2.0**

Additional terms: All images generated using AI-VIS that are made publicly available must be marked as AI-VIS generated to avoid confusion with real visible imagery.

Paper: [Simulating Nighttime Visible Satellite Imagery of Tropical Cyclones Using Conditional Generative Adversarial Networks](https://ieeexplore.ieee.org/document/10988561)

```bibtex
@ARTICLE{10988561,
  author={Yao, Jinghuai and Du, Puyuan and Zhao, Yucheng and Wang, Yubo},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Simulating Nighttime Visible Satellite Imagery of Tropical Cyclones Using Conditional Generative Adversarial Networks}, 
  year={2025},
  volume={18},
  number={},
  pages={12616-12633},
  keywords={Clouds;Monitoring;Cloud computing;Data models;Tropical cyclones;Spatial resolution;Satellites;Satellite broadcasting;Loss measurement;Earth;Advanced Himawari imager (AHI);clouds;conditional generative adversarial network (CGAN);deep learning;nighttime;tropical cyclone (TC);visible (VIS)},
  doi={10.1109/JSTARS.2025.3567074}}
```

Dataset: [HuggingFace: Dapiya/aivis-dataset](https://huggingface.co/datasets/Dapiya/aivis-dataset)
(Uploading)

Training Code: [GitHub: Dapiya/aivis-training](https://github.com/Dapiya/aivis-training) (Support missing, not guaranteed to work, contact us if you have questions)

# Test Scripts Provided

`test_aivis.py` is a demo script for Himawari-8/9 target area data.

`test_aivis_fldk.py` is for full disk data.

`test_aivis_floater.py` does floater imagery

# Models

| Model Name | Params* | Training Finish Time | Weights |
|------------|--------|---------------------| -------|
| aivis-1.0  |  67M   | 2024/3 | [HF🤗](https://huggingface.co/Dapiya/aivis-1.0) |
| aivis-1.5-small  |  67M   | 2024/9 | [Request form](https://docs.google.com/forms/d/1dBqFUJSB15ZhTCaj-W_WARUyOnBhL5cQANw7tEZhgeo) |
| aivis-1.5-large  |  263M   | 2024/12 | [Request form](https://docs.google.com/forms/d/1dBqFUJSB15ZhTCaj-W_WARUyOnBhL5cQANw7tEZhgeo) |

Upscaler 1.5: [HF🤗](https://huggingface.co/Dapiya/aivis-upscaler-1.5)

If you're unable to access Google Forms, please email wang3399@wisc.edu

*Params are counting the generator only, as only the generator is used during inference, and the discriminator is very small compared to the generator.

# Hardware Requirements

AI-VIS runs on any relatively modern Nvidia GPU with at least 3GB of VRAM. Typical forward pass of a single pass takes <0.1s(0.048s tested on RTX2080 Ti), other components of the pipeline takes significantly longer than running the model itself.

All code in this repo has been tested with RTX 20, 30, and 40 series GPUs

Usage with CPU(may have performance issue) has not been well tested, should work with no or minimum modification.

Usage with AMD GPUs are untested for now, again, should work with no or minimum modification. Tell us your experience if you try it.

## TPU (Google Cloud TPU / PyTorch/XLA)

- Status: Experimental inference support via PyTorch/XLA.
- Requirements: Install a matching `torch_xla` build for your PyTorch version on a TPU VM or Colab TPU runtime. See the official PyTorch/XLA install docs.
- Scope: The main AI‑VIS UNet runs on TPU. The optional Real‑ESRGAN upscaler continues to run on CPU/GPU.

Example usage:

```python
from aivis.aivis import AI_VIS

# Explicitly request TPU/XLA device
model = AI_VIS(device='xla')  # or 'tpu'
model.load(weight_path='./aivis/weights', arch='1.5-large')

# ... run inference the same as GPU/CPU ...
```

Tips for TPU runtimes:
- Use `XLA_USE_BF16=1` to prefer bfloat16 for speed/VRAM when appropriate.
- Marking steps is handled automatically during forward; no extra code needed.
- If you enable `--upscale`, the upscaler will fall back to CPU/GPU; TPU execution for upscaler is not supported.

# Usage

1. Clone the repository

```bash
git clone github.com/Dapiya/AI-VIS.git
```

2. Install corresponding version of PyTorch from [PyTorch website](https://pytorch.org/)

3. Install the requirements (python 3.10/11, support for 3.12 is not guaranteed)

```bash
pip install -r requirements.txt
```

4. Download the weights and place in ./aivis/weights
(see table above for links)

5. (Optional) Download data from [AWS](https://noaa-himawari9.s3.amazonaws.com/index.html#AHI-L1b-Target/) and place in ./aivis/test_data/HIMAWARI

    Channels 8, 9, 10, 11, 13, 15, 16 are needed

Note: The package already includes a sample of data. If you want to test with anything else, replace it with the data you downloaded.

6. Run the inference script

```bash
python test_aivis.py [--upscale] [--half-precision] [--device cpu|cuda|xla]
```

Note: Upscaler model must be downloaded and placed into ./aivis/weights folder when doing --upscale

**`test_aivis_fldk.py`** and **`test_aivis_floater.py`** are similar to **`test_aivis.py`**, use `--help` to see the options.

`test_aivis_fldk.py` has a `--fake-time` option that sets the time of the fldk to the specified time in UTC on Mar 21(equinox), note that if the time entered produces nighttime in certain parts of the full disk, there might be unexpected results(but not a corrupted image). This option is purely so that people know this could be done, feel free to change the code to whatever you want to try out.

# Future Plans

- We are working on stuff related to cross-satellite calibration to improve the performance of AI-VIS, since it is only trained on Himawari-8 data. Our previous experiments suggested significant degradations. We will roll out a complete set of system along with support for GOES-R series in the future.
- In the slightly further future, we will begin working on the next generation of AI-VIS, aiming to continue improving the capabilities of the model.
