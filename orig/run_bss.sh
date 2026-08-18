git clone https://github.com/nttcslab/m2d.git \
cd m2d \
curl -o util/lars.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/lars.py
curl -o util/lr_decay.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/lr_decay.py
curl -o util/lr_sched.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/lr_sched.py
curl -o util/misc.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/misc.py
curl -o util/analyze_repr.py https://raw.githubusercontent.com/daisukelab/general-learning/master/SSL/analyze_repr.py
curl -o m2d/pos_embed.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py
curl -o train_audio.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/main_pretrain.py
curl -o speech/train_speech.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/main_pretrain.py
curl -o audioset/train_as.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/main_pretrain.py
curl -o clap/train_clap.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/main_pretrain.py
curl -o mae_train_audio.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/main_pretrain.py
curl -o m2d/engine_pretrain_m2d.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/engine_pretrain.py
curl -o m2d/models_mae.py https://raw.githubusercontent.com/facebookresearch/mae/efb2a8062c206524e35e47d04501ed4f544c0ae8/models_mae.py
curl -o m2d/timm_layers_pos_embed.py https://raw.githubusercontent.com/huggingface/pytorch-image-models/e9373b1b925b2546706d78d25294de596bad4bfe/timm/layers/pos_embed.py
patch -p1 < patch_m2d.diff

conda create -n bss python
pip install timm einops nnAudio librosa >& /dev/null \
pip install -r requirements.txt
mv UrbanSound ./
mv generate_labels.py ./
python generate_labels.py

conda install python=3.8
git clone https://github.com/Audio-AGI/AudioSep.git && \
cd AudioSep && \ 
mkdir checkpoint, output
cd checkpoint
curl -o audiosep_base_4M_steps.ckpt https://huggingface.co/spaces/Audio-AGI/AudioSep/resolve/main/checkpoint/audiosep_base_4M_steps.ckpt?download=true
curl -o music_speech_audioset_epoch_15_esc_89.98.pt https://huggingface.co/spaces/Audio-AGI/AudioSep/resolve/main/checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt?download=true
python3.8 -m pip install torchlibrosa==0.1.0 gradio==3.47.1 gdown lightning transformers==4.28.1 ftfy braceexpand webdataset soundfile wget h5py
python3.8 -m pip install torch torchaudio torchvision
cd ..
mv ../../bss.py ./
python3 bss.py