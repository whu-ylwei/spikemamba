## README 

## Training:
To train the model, you need to set some esstential parameters in the config file. Config files should be put in the `./config/` folder. In addition, the training dataset is a folder named 'DENSE', and please set parameter `--datafolder` in command to set the root folder of the dataset 'DENSE'. In the config file You alse may pay attention to the `save_dir` setting that saves all checkpoints of the model and detailed information during training.

```bash
CUDA_VISIBLE_DEVICES=0,1 python train_parallel.py --config configs/train_s2d_spiketransformer.json --datafolder ... --multiprocessing_distributed
```


## Testing:
### Test script
Testing a network is done by using two scripts. First, the `test_DENSE.py` script is used to save the outputs of the network.
As a second step, the `evaluation_DENSE.py` script is used to calculate the metrics based on these outputs.

To run the `test_DENSE.py` script, `'--path_to_model'` is the model path that you want to evaluate, `'--output_path'` is the folder that saves all the testing results including files with .npy and .png formats ( these files are used in the `evaluation_DENSE.py` script), and you alse need to set the `'--data_folder'` to set the root folder of the testing dataset.

For example:

```bash
CUDA_VISIBLE_DEVICES=0 python test_DENSE.py --path_to_model .../model_best.pth.tar --output_path ... --data_folder .../DENSE/test/
```


### Evaluation script
To run the `evaluation_DENSE.py` script, three parameters is important. Specifically, `'--target_dataset'` has to be specified as `.../ground_truth/npy/depth_image/`,  `'--predictions_dataset'` has to be specified as `.../npy/image/`, and `'--output_folder'` can be set to save the final visualization results.

For example:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation_DENSE.py --target_dataset .../ground_truth/npy/depth_image/ --predictions_dataset .../npy/image/ --clip_distance 1000 --reg_factor 5.7
```


## Environment
To help successfully run the code, the esstential environment on Linux is included in the `requirements.txt`.

## Mamba-SSM Backend Setup

The 3D encoder now supports `mamba_ssm` as the Mamba core backend.

1. Install dependencies:

```bash
pip install -r requirements.server.txt
pip install mamba-ssm causal-conv1d
```

2. Configure `config["model"]`:

```json
{
  "mamba_backend": "mamba_ssm",
  "mamba_require_ssm": true,
  "mamba_d_state": 16,
  "mamba_d_conv": 4,
  "mamba_expand": 2
}
```

3. If installation fails because of torch/cuda mismatch, set:

```json
{
  "mamba_backend": "fallback",
  "mamba_require_ssm": false
}
```

Then train normally and switch back to `mamba_ssm` after fixing the environment.



## Acknowledgement
The structure of the codebase is borrowed from [RAMNet](https://rpg.ifi.uzh.ch/RAMNet.html) and the base of the encoder backbone is borrowed from  [Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer)
