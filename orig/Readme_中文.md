运行run_bss.sh
环境要求：
- linux/ios环境
- 下载urban sound数据：https://urbansounddataset.weebly.com/urbansound.html
放在项目文件夹bss/urbansound/下面，用generate_dataset.py生成混合音频。
分离的音源在bss/m2d/AudioSep/outpu里面看。

conda activate m2d --python3.7
cd Documents/bss_v0/m2d
python generate_labels.py

conda activate bss --python3.11
cd AudioSep
python bss.py