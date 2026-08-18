
## 1. Setup

- Deployable for all os systems, but better for linux/ios.
- Download the dataset of urban sounds ：https://urbansounddataset.weebly.com/urbansound.html or http://serv.cusp.nyu.edu/projects/urbansounddataset.
and save to the folder: `data/UrbanSound/audios/`，and use generate_dataset.py to generate mixed audios.
- Format the code for the experiments
- Integrity check for the data
- Run the command
```shell
cd MAE-RNET
bash run_bss.sh
```
- Check the seperated audios in the folder of `output/`, and the evaluation results in the folder of `evaluation/`.


Major folders

You will find the following folders after the setup.
- config     -- YAML files configuring the settings of tagging and separator two engines.
- data/UrbanSound     -- The data used during the training and testing
- /audios -- Original source audios of 10 urban sound types.
- /metadata.csv  -- The metadata file to list the generated mixed audio and the used source audios.
- tagging  -- The agent to classify the source audios from the loaded mixed audios with MAE based M2D pretrained framework.
- separator  -- The agent to separate the audio with ResUNet30.
- output   -- The separated audios from the mixed audios.
- evaluation -- The evaluation metrics between source audios constructing the mixed audios with the separated audios from the mixed audios.


## 2. Running Experiments and Summarizing the results

conda activate bss --python3.11
cd MAE-RNET
python bss.py


- `data_dir`: type=str,default='./data/UrbanSound/mixed/', 'directory of the original data.'  
- `metadata_path`,type=str,default='./data/UrbanSound/metadata.csv', 'path of the metadata file of the generated mixed audio and the used source audios.'
- `tagging_path`,type=str,default='./tagging/representation_result.csv', 'path of generated tags for each input mixed audio.'
- `graph_dir`,type=str,default='./graph/', 'directory of graphs'
- `output_dir`,type=str,default='./output/', 'directory of outputs for bss task.'
- `evaluation_dir`,type=str,default='./evaluation/', 'directory of the evaluation metrics between source audios constructing the mixed audios with the seperated audios from the mixed audios.'
- `log_dir`,type=str,default='./log/', 'directory of the transaction logs.'
- `noise_factor`,type=float,default=0, 'noise to be added to original files for experiments.'

## Acknowledgements

This code is based on "[Masked Modeling Duo (M2D) & M2D-CLAP](https://github.com/nttcslab/m2d)" and "[AudioSep](https://github.com/Audio-AGI/AudioSep.git)".

Data:
["J. Salamon, C. Jacoby and J. P. Bello, A Dataset and Taxonomy for Urban Sound Research, 22nd ACM International Conference on Multimedia, Orlando USA, Nov. 2014.](http://serv.cusp.nyu.edu/projects/urbansounddataset)".

These publicly available implementations and open data for the experiments are sincerely appreciated.

```bibtex
@inproceedings{niizumi2023m2d,
    title   = {{Masked Modeling Duo: Learning Representations by Encouraging Both Networks to Model the Input}},
    author  = {Daisuke Niizumi and Daiki Takeuchi and Yasunori Ohishi and Noboru Harada and Kunio Kashino},
    booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
    year    = {2023},
    url     = {https://ieeexplore.ieee.org/document/10097236},
    doi     = {10.1109/ICASSP49357.2023.10097236}}

@article{liu2023separate,
  title={Separate Anything You Describe},
  author={Liu, Xubo and Kong, Qiuqiang and Zhao, Yan and Liu, Haohe and Yuan, Yi, and Liu, Yuzhuo, and Xia, Rui and Wang, Yuxuan, and Plumbley, Mark D and Wang, Wenwu},
  journal={arXiv preprint arXiv:2308.05037},
  year={2023}
}
```

