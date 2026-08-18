import torch

import wget
import pandas as pd
import os
import librosa
import difflib
import numpy as np
import fast_bss_eval
from scipy.io import wavfile
import warnings; warnings.simplefilter('ignore')
import logging; logging.basicConfig(level=logging.INFO)
from pathlib import Path
import zipfile
import gc
from IPython.display import display, Audio
from sentence_transformers import SentenceTransformer
import tagging
from tagging.portable_m2d import PortableM2D
import separator
from separator.pipeline import build_audiosep, separate_audio
import utils
import argparse
import yaml
import fast_bss_eval
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class tag():
      def config():
            with open ('config/tagging_base.yaml','r') as f:
                  config_tag = yaml.safe_load(f)
            sr = config_tag['data']['sample_rate']
            duration = config_tag['data']['duration']
            min_duration = config_tag['data']['min_duration']
            target_sr = config_tag['data']['target_rate']
            lower_db = config_tag['data']['loudness_norm']['lower_db']
            higher_db = config_tag['data']['loudness_norm']['higher_db']
            k = config_tag['data']['number_of_top_classes']
            hop = config_tag['data']['hop']
            return sr, duration ,min_duration ,target_sr ,lower_db ,higher_db ,k ,hop
      def show_topk(classes, sr, target_sr ,k , m2d, wav_file, model_similarity):
            #gc.collect()

            #print(wav_file)
            # Loads and shows an audio clip.
            wav, sr = librosa.load(wav_file, mono=True, sr=sr)
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            
            wav = torch.tensor(wav).unsqueeze(0)
            # Predicts class probabilities for the batch segments.

            with torch.no_grad():
                  try:
                        probs = m2d(wav).squeeze(0).softmax(0)
                  except Exception as error:
                        print(error)
                        with open ("error.log", "a") as f:
                              f.write(str(error))
                              f.close()
                        probs = []
            # Shows the top-k prediction results.
            topk_values, topk_indices = probs.topk(k=k)
            print(topk_values[0])

            topk_classes = [classes.loc[i].display_name for i, v in zip(topk_indices.numpy(), topk_values.numpy())]
            print(topk_classes)
            wav_ = str(os.path.dirname(wav_file))+"\\"+str(os.path.basename(wav_file))
            print(wav_)
            print(', '.join([f'{classes.loc[i].display_name} ({v*100:.1f}%)' for i, v in zip(topk_indices.numpy(), topk_values.numpy())]))
            # Shows the top-k prediction results.
            sub_representation_ = pd.DataFrame()
                         
            #print(topk_classes)

            sub_representation_['classes']=topk_classes
            sub_representation_['values']=topk_values
            sub_representation_['wav_name']=wav_


            for c1 in topk_classes:
                  for c2 in topk_classes:
                        if 'outside' in c2.lower() :
                              topk_classes.remove(c2)
                        elif 'noise' in c2.lower():
                              topk_classes.remove(c2)
                        else:
                              similarity_score = utils.utils.similarity(c1,c2,model_similarity)
                              if (similarity_score>0.8 and similarity_score<1)==True:
                                    print(c1,c2,similarity_score)
                                    topk_classes.remove(c2)
            print(len(topk_classes))
            print(topk_classes)
            #print(dir_)
            filtered = pd.DataFrame()
            filtered['classes']=topk_classes

            sub_representation = pd.merge(filtered,sub_representation_,how='inner',on='classes')

            #sub_representation = sub_representation[['wav_name','classes']]
            
            print(sub_representation)
            
            return topk_classes, topk_values,  sub_representation

      def show_topk_sliding_window(classes, sr, duration ,min_duration ,target_sr ,lower_db ,higher_db ,k ,hop,m2d, wav_file,opt):
            gc.collect()
            #64000-2,64000-4,16000
            #print(wav_file)
            noise_factor = opt.noise_factor
            print(m2d.cfg.sample_rate)
            gc.collect()
            wav, sr = librosa.load(wav_file, mono=True, sr=sr)      
            average_wav = np.median(wav)  
            noise = np.random.uniform(low=lower_db, high=higher_db, size=len(wav))*average_wav*noise_factor
            #print(wav)
            #print(noise.shape)
            wav = (wav+noise).astype(np.float32)


            # Loads and shows an audio clip.
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            # Makes a batch of short segments of the wav into wavs, cropped by the sliding window of [hop, duration].
            wavs = [wav[int(c * sr) : int((c + duration) * sr)] for c in np.arange(0, wav.shape[-1] / sr, hop)]
            wavs = [utils.utils.repeat_if_short(wav,min_duration) for wav in wavs]

            wavs = torch.tensor(wavs)
            wav_ = str(os.path.dirname(wav_file))+"\\"+str(os.path.basename(wav_file))
            # Predicts class probabilities for the batch segments.
            with torch.no_grad():
                  try:
                        probs_per_chunk = m2d(wavs).softmax(1)
                  except Exception as error:
                        print(error)
                        with open (log_dir+"error.log", "a") as f:
                              f.write(str(error))
                              f.close()
            # Shows the top-k prediction results.
            topk_classes=[]
            top_values=[]
            secs=[]
            sub_representation = pd.DataFrame()
            for i, probs in enumerate(probs_per_chunk):
                  topk_values, topk_indices = probs.topk(k=k)
                  sec = f'{i * hop:d}s '
                  print(sec, ', '.join([f'{classes.loc[i].display_name} ({v*100:.1f}%)' for i, v in zip(topk_indices.numpy(), topk_values.numpy())]))
                  for i, v in zip(topk_indices.numpy(), topk_values.numpy()):
                        topk_classes.append(classes.loc[i].display_name )
                        top_values.append(v)
                        secs.append(sec)
                        

            
            #print(topk_classes)
            wav_name = str(wav_file).split('\\')[-1]
            
            sub_representation['time_frame']=secs
            sub_representation['classes']=topk_classes
            sub_representation['values']=top_values
            sub_representation['wav_name']=wav_
            
            sub_representation['time_frame_int']=sub_representation['time_frame'].str.replace('s','')
            sub_representation['time_frame_int']=sub_representation['time_frame_int'].astype(int)
            sub_representation = sub_representation.sort_values(by='time_frame_int').reset_index()
            sub_representation = sub_representation[['wav_name','time_frame','classes','values']]
            
            #print(sub_representation)
            
            return topk_classes, top_values, secs, sub_representation
      
      def show_topk_for_all_frames(classes, sr, duration ,min_duration ,target_sr ,lower_db ,higher_db ,k ,hop,m2d, wav_file,opt):
            gc.collect()
            #print(wav_file)
            print(m2d.cfg.sample_rate)
            # Loads and shows an audio clip.
            wav, sr = librosa.load(wav_file, mono=True, sr=sr)      
            noise_factor = opt.noise_factor
            print(m2d.cfg.sample_rate)
            average_wav = np.median(wav)  
            noise = np.random.uniform(low=-1.0, high=1.0, size=len(wav))*average_wav*noise_factor
            #print(wav)
            #print(noise.shape)
            wav = (wav+noise).astype(np.float32)
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            display(Audio(wav, rate=m2d.cfg.sample_rate))
            wav = torch.tensor(wav)
            wav_ = str(os.path.dirname(wav_file))+"/"+str(os.path.basename(wav_file))
            # Predicts class probabilities for all frames.
            with torch.no_grad():
                  logits_per_chunk, timestamps = m2d.forward_frames(wav.unsqueeze(0))  # logits_per_chunk: [1, 62, 527], timestamps: [1, 62]
                  probs_per_chunk = logits_per_chunk.squeeze(0).softmax(-1)  # logits [1, 62, 527] -> probabilities [62, 527]
                  timestamps = timestamps[0]  # [1, 62] -> [62]
            # Shows the top-k prediction results.
            time_frame=[]
            topk_classes=[]
            top_values=[]
            secs=[]
            sub_representation = pd.DataFrame()
            for i, (probs, ts) in enumerate(zip(probs_per_chunk, timestamps)):
                  topk_values, topk_indices = probs.topk(k=k)
                  topk_classes = [classes.loc[i].display_name for i, v in zip(topk_indices.numpy(), topk_values.numpy())if v>1]
                  print('topk_classes are: ',topk_classes)
                  sec = f'{ts/1000:.1f}s '
                  print(sec, ', '.join([f'{classes.loc[i].display_name} ({v*100:.1f}%)' for i, v in zip(topk_indices.numpy(), topk_values.numpy()) if v>1]))
                  for i, v in zip(topk_indices.numpy(), topk_values.numpy()):
                        topk_classes.append(classes.loc[i].display_name )
                        top_values.append(v)
                        secs.append(sec)
                  

            
            print(topk_classes)
            wav_name = str(wav_file).split('\\')[-1]
            
            sub_representation['time_frame']=secs
            sub_representation['classes']=topk_classes
            sub_representation['values']=top_values
            sub_representation['wav_name']=wav_
            
            sub_representation['time_frame_int']=sub_representation['time_frame'].str.replace('s','')
            sub_representation['time_frame_int']=sub_representation['time_frame_int'].astype(int)
            sub_representation = sub_representation.sort_values(by='time_frame_int').reset_index()
            sub_representation = sub_representation[['wav_name','time_frame','classes','values']]
            
            print(sub_representation)
            
            return topk_classes, top_values, secs, sub_representation

class bss():
      def loadData(tagging_path):
      # AudioSep processes the audio at 32 kHz sampling rate
            df_representation = pd.read_csv(tagging_path)
            audio_files = df_representation ["wav_name"]
            topk_classes = df_representation ["classes"]
            return audio_files,topk_classes

      def audioSeparate(model, audio_files,topk_classes,output_dir):
            #gc.collect()
            for i in range(len(audio_files)):
                  #time.sleep(5)
                  audio_file=audio_files[i]
                  #print(audio_file)
                  wav_name = os.path.basename(audio_file)
                  texts = topk_classes[i]
                  text=texts.replace("'","")
                  texts =texts.replace("[","")
                  texts =texts.replace("]","")
                  texts =texts.split(",")
                  print(texts)
                  for text in texts:
                        text = text.lower()
                        text = text.replace("'","")
                        print(text)
                        output_file=output_dir+wav_name+"_"+text+'.wav'
                        separate_audio(model, audio_file, text, output_file, device)
            return output_file
class eval():
      def bssEval(source_file,output_file,sr):
            ref,fs = librosa.load(source_file, mono=True, sr=sr)
            print(ref.shape)
            est,_ = librosa.load(output_file, mono=True, sr=sr)
            print(est.shape)
            if len(est)>len(ref):
                  est = est[:ref.shape[0]]
            else:
                  ref = ref[:est.shape[0]]
            print(ref.shape)
            print(np.shape(est))
            # compute all the metrics
            #sdr, sisdr , sar, perm = fast_bss_eval.bss_eval_sources(ref.T,est.T)
            sdr= utils.calculate_sdr(ref, est,eps=1e-10)
            sisdr = utils.calculate_sisdr(ref, est)
            mse = utils.calculate_mse(ref, est)
            print(sdr, sisdr )
            #print(sdr, sisdr, sar, perm )
            return sdr, sisdr, mse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str,default='data/DCASE2025Task4EvaluationDataset/eval_set/test/', help = 'directory of the original data.' ) 
    parser.add_argument('--metadata_path',type=str,default='data/DCASE2025Task4EvaluationDataset/eval_set/meta.csv', help = 'path of the metadata file of the generated mixed audio and the used source audios.')
    parser.add_argument('--tagging_path',type=str,default='tagging/dcase2025/representation_result.csv', help = 'path of generated tags for each input mixed audio.')
    parser.add_argument('--graph_dir',type=str,default='graph/dcase2025/', help = 'directory of graphs.' )
    parser.add_argument('--output_dir',type=str,default='output/dcase2025/', help = 'directory of outputs for bss task.')
    parser.add_argument('--log_dir',type=str,default='log/dcase2025/', help = 'directory of the transaction logs.')
    parser.add_argument('--evaluation_dir',type=str,default='evaluation/dcase2025/', help = 'directory of the evaluation metrics between source audios constructing the mixed audios with the seperated audios from the mixed audios.')
    parser.add_argument('--noise_factor',type=float,default=0, help = 'noise to be added to original files for experiments')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
      #data_dir = 'data/UrbanSound/mixed/'
      #metadata_path = 'data/UrbanSound/metadata.csv'
      #project_path = "../"    
      gc.collect()
      project_dir=os.getcwd()
      os.chdir(project_dir)
      opt = get_parser()
      data_dir = opt.data_dir
      metadata_path = opt.metadata_path
      tagging_path = opt.tagging_path
      output_dir = opt.output_dir
      graph_dir = opt.graph_dir
      evaluation_dir = opt.evaluation_dir
      log_dir = opt.log_dir
      noise_factor = opt.noise_factor
      if os.path.exists(graph_dir)==False:
            os.makedirs(graph_dir)
      if os.path.exists(output_dir)==False:
            os.makedirs(output_dir)
      if os.path.exists(log_dir)==False:
            os.makedirs(log_dir)
      #modeltagging = PortableM2D(weight_file='./tagging/m2d_vit_base-80x1001p16x16-221006-mr7_as_46ab246d/weights_ep69it3124-0.47929.pth', num_classes=527)
      model_tagging = PortableM2D(weight_file='tagging/m2d_clap_vit_base-80x1001p16x16-240128_AS-FT_enconly/weights_ep67it3124-0.48558.pth',num_classes=527)
      model_similarity = SentenceTransformer('all-mpnet-base-v2')
      classes = pd.read_csv('class_labels_indices.csv').sort_values('mid').reset_index()
      
      files = list(Path(data_dir).glob('*.wav'))
      files = np.random.choice(files, size=len(files), replace=False)
      sr, duration ,min_duration ,target_sr ,lower_db ,higher_db ,k ,hop= tag.config()
      '''
      wav_name =[]
      topk_=[]
      df_representation = pd.DataFrame()
      try:
            for fn in files:
                  topk_classes, topk_values, sub_representation=tag.show_topk(classes, sr, target_sr ,k , model_tagging, fn, model_similarity)
                  print(topk_classes)
                  df_representation = pd.concat([df_representation,sub_representation],axis=0,ignore_index=True )
                  
                  time.sleep(5)

      except Exception as error:
            print(error)
            with open (log_dir+"error.log", "a") as f:
                  f.write(str(error))
                  f.close()
      df_representation.to_csv(tagging_path)
      model = build_audiosep(
            config_yaml='config/bss_base.yaml',
            checkpoint_path="separator/checkpoint/audiosep_base_4M_steps.ckpt",
            device=device)
      audio_files,topk_classes = bss.loadData(tagging_path)
      output_file = bss.audioSeparate(model,audio_files,topk_classes,output_dir)
      '''
      metadata = pd.read_csv(metadata_path)
      metadata['included']=0
      for i in range(len(metadata)):
            source_file_mixed = metadata["combined"][i]
            print(len(files))
            for f in files:
                  f = str(f).split('\\')[-1]
                  if str(f) in str(source_file_mixed):
                        metadata.loc[i,'included']=1


      metadata = metadata[metadata['included']==1]
      metadata=metadata.sort_values('combined').reset_index()
      metadata = metadata.drop(['included','index'],axis=1)
      print(metadata)
      
      source_names=[]
      source_classes=[]
      source_files=[]
      source_files_mixed=[]
      for i in range(len(metadata)):
            source_file_mixed = metadata["combined"][i]
            for col in metadata.columns:
                  if 'source' in col:
                        if 'class' not in col:
                              if type(metadata[col][i])==str:
                                    print( metadata[col][i])
                                    #source_name = metadata[col][i].split('/')[-2]
                                    source_name = metadata[col][i].split('/')[-1].split('.')[0].split('_')[-1]
                                    source_names.append(source_name)
                                    source_file = metadata[col][i]
                                    source_files.append(source_file)
                                    source_files_mixed.append(source_file_mixed)  

                        #else:
                              #source_name = metadata[col][i]
                              #source_names.append(source_name)
                       

      #print(source_names,source_files,source_files_mixed)  
      df_source = pd.DataFrame()
      df_source['source_names']=source_names
      df_source['source_files']=source_files
      df_source['source_files_mixed']=source_files_mixed

      print(df_source)
      df_output = pd.DataFrame()
      estimated_names=[]
      output_files=[]
      output_dirs = []
      for r,d,f in os.walk(output_dir):
            for f_ in f:
                  estimated_name = f_.split('.wav_')[1].split('.')[0]
                  output_files.append(f_)
                  output_dirs.append(r)
                  estimated_names.append(estimated_name)
      df_output['estimated_names']=estimated_names
      df_output['output_files']=output_files
      df_output['output_dirs']=output_dirs

      print(df_output)
      
      for i in range(len(df_source)):
            original_mixed = df_source['source_files_mixed'][i]
            source_file_name = original_mixed.split('/')[-1]
            source_name = df_source['source_names'][i]
            source_file = df_source['source_files'][i]
            if 'children_playing' in source_name:
                  source_name = 'children_speech'
            for j in range(len(df_output)):
                  output_file_name = df_output['output_files'][j]
                  estimated_name = df_output['estimated_names'][j]
                  similarity_score = utils.utils.similarity(source_name,estimated_name,model_similarity)
                  print(similarity_score,source_name,estimated_name )
                  if (source_file_name in output_file_name and similarity_score>0.4 )==True:
                        output_file = df_output['output_dirs'][j]+"/"+output_file_name
                        print(source_file_name, output_file_name, source_file,output_file )
                        sdr, sisdr, mse  = eval.bssEval( source_file,output_file,sr)
                        bss_eval = (sdr, sisdr, mse )
                        with open (evaluation_dir+"bss_eval.log","a") as f:
                              f.write("source_file: {}, output_file is: {}, bss_eval_result is: {}".format(source_file,output_file,bss_eval)+"\n")
                              f.close()
                  j+=1
            i+=1



