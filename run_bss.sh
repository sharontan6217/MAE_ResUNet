git clone https://github.com/sharontan6217/MAE-RNet.git && \
cd MAE_RNet/tagging && \
curl -o m2d_clap_vit_base-80x1001p16x16-240128_AS-FT_enconly https://github.com/nttcslab/m2d/releases/download/v0.3.0/m2d_clap_vit_base-80x1001p16x16-240128_AS-FT_enconly.zip && \
curl -o m2d_vit_base-80x1001p16x16-221006-mr7_as_46ab246d https://github.com/nttcslab/m2d/releases/download/v0.3.0/m2d_vit_base-80x1001p16x16-221006-mr7_as_46ab246d.zip && \ 
conda create -n bss python=3.11 && \
pip install -r requirements.txt && \
cd ../seperator/checkpoint && \
curl -o audiosep_base_4M_steps.ckpt https://huggingface.co/spaces/Audio-AGI/AudioSep/resolve/main/checkpoint/audiosep_base_4M_steps.ckpt?download=true && \
cd ../.. && \
python3 bss.py