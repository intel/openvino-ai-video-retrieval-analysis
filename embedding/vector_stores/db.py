#Copyright (C) 2024 Intel Corporation
#SPDX-License-Identifier: Apache-2.0

from langchain_community.vectorstores import VDMS
from langchain_community.vectorstores.vdms import VDMS_Client
from pydantic.v1 import BaseModel,root_validator
#from langchain.pydantic_v1 import BaseModel, root_validator
from langchain_core.embeddings import Embeddings
from decord import VideoReader, cpu, gpu
import numpy as np
from typing import List, Optional, Dict, Any

from dateparser.search import search_dates
import datetime

from embedding.meanclip_modeling.simple_tokenizer import SimpleTokenizer
from embedding.meanclip_datasets.preprocess import get_transforms
from einops import rearrange


import torch
import os
import time
import cv2




import torchvision.transforms as T
toPIL = T.ToPILImage()



from openvino.runtime import Core

import openvino as ov
import os


def transform(image, n_px):
    """
    Preprocessing pipeline equivalent to the PyTorch _transform function.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        n_px (int): Target size for resizing and cropping.

    """
    # 1. Resize with bicubic interpolation
    image = cv2.resize(image, (n_px, n_px), interpolation=cv2.INTER_CUBIC)
    

 
    return torch.from_numpy(image)

class OpenVINOMeanClip:
    def __init__(self, model_path: str, device: str = "CPU"):
        self.core = Core()

        if "NPU" in device:
            with open("MeanCLIP.blob", "rb") as f:
                self.compiled_model = self.core.import_model(f.read(), device)        
        else:
            self.model = self.core.read_model(model_path)
            self.compiled_model = self.core.compile_model(self.model, device)
            
        self.output_blob = self.compiled_model.output(0)
        
    def get_video_embeddings(self, video_tensor: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(self.compiled_model(video_tensor)[self.output_blob]).clone()
        
        
        
# 'similarity', 'similarity_score_threshold' (needs threshold), 'mmr'

g_meanClip = None
vpath = None
vr_global = None

class MeanCLIPEmbeddings(BaseModel, Embeddings):
    """MeanCLIP Embeddings model."""

    model: Any
    preprocess: Any
    tokenizer: Any
    ov_device: str = "GPU"
    # Select model: https://github.com/mlfoundations/open_clip
    model_name: str = "ViT-H-14"
    checkpoint: str = "laion2b_s32b_b79k"
    
    

    @root_validator(allow_reuse=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that open_clip and torch libraries are installed."""
        try:
            # Use the provided model if present
            if "model" not in values:
                raise ValueError("Model must be provided during initialization.")
            values["preprocess"] = get_transforms
            values["tokenizer"] = SimpleTokenizer()

        except ImportError:
            raise ImportError(
                "Please ensure CLIP model is loaded"
            )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        model_device = next(self.model.clip.parameters()).device
        text_features = []
        for text in texts:
            # Tokenize the text
            if isinstance(text, str):
                text = [text]

            sot_token = self.tokenizer.encoder["<|startoftext|>"]
            eot_token = self.tokenizer.encoder["<|endoftext|>"]
            tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]
            tokenized_text = torch.zeros((len(tokens), 64), dtype=torch.int64)
            for i in range(len(tokens)):
                if len(tokens[i]) > 64:
                    tokens[i] = tokens[i][:64-1] + tokens[i][-1:]
                tokenized_text[i, :len(tokens[i])] = torch.tensor(tokens[i])

            text_embd, word_embd = self.model.get_text_output(tokenized_text.unsqueeze(0).to(model_device), return_hidden=False)

            # Normalize the embeddings
  
            text_embd = rearrange(text_embd, "b n d -> (b n) d")
            text_embd = text_embd / text_embd.norm(dim=-1, keepdim=True)

            # Convert normalized tensor to list and add to the text_features list
            embeddings_list = text_embd.squeeze(0).tolist()
     
            text_features.append(embeddings_list)

        return text_features


    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


    def embed_video(self, paths: List[str], **kwargs: Any) -> List[List[float]]:
        global g_meanClip
        


        if os.path.isfile("MeanCLIP.xml"):
            pass
        else:
            import subprocess
            from compile_model_npu import compile_and_export_model
            print("MeanClip OV model doesn't exist, converting now. Please wait this step will take some time as NPU compilation takes time.")
            for vid_path in sorted(paths):
          
                # Encode the video to get the embeddings
                model_device = next(self.model.parameters()).device
                # Preprocess the video for the model
            
            
                videos_tensor= self.load_video_for_meanclip(vid_path, num_frm=self.model.num_frm,
                                                                                max_img_size=224,
                                                                                start_time=kwargs.get("start_time", None),
                                                                                clip_duration=kwargs.get("clip_duration", None)
                                                                            )
            
                
                input_tensor = videos_tensor.to(model_device)
                #input_tensor = videos_tensor.unsqueeze(0).to(model_device)
                
                
                print("Step 1: Converting Pytorch model to onnx")
                torch.onnx.export(self.model, (input_tensor), "MeanCLIP.onnx", input_names=["videos"], output_names=["embeddings"])
                print("onnx model genertated")
                print("Step 2: Converting Onnx model to OV")
                subprocess.call(["mo", '--input_model', "MeanCLIP.onnx"])
                print("openvino model genertated")
                model_path = "MeanCLIP.xml"  
        if "NPU" in self.ov_device:
            if os.path.isfile("MeanCLIP.blob"):
                pass
            else:
                print("Step 3: Compiling blob for NPU. This is a one time step and will take about 30 min if no blob is present")
                output_path = "MeanCLIP.blob"   
                compile_and_export_model(ov.Core(),model_path,output_path)                

                    #import sys
                    #sys.exit(1)         
        if g_meanClip is None:
            print("creating OpenVINOMeanClip to run on:",self.ov_device)
            g_meanClip = OpenVINOMeanClip("MeanCLIP.xml",self.ov_device)
            
        
        # Open images directly as PIL images

        video_features = []
        for vid_path in sorted(paths):
            # Encode the video to get the embeddings
            model_device = next(self.model.parameters()).device
            # Preprocess the video for the model
            
            t0 = time.time()
            videos_tensor= self.load_video_for_meanclip(vid_path, num_frm=self.model.num_frm,
                                                                              max_img_size=224,
                                                                              start_time=kwargs.get("start_time", None),
                                                                              clip_duration=kwargs.get("clip_duration", None)
                                                                          )
            t1 = time.time()
            print("self.load_video_for_meanclip took ", t1 - t0, " s")
            #input_tensor = videos_tensor.unsqueeze(0).to(model_device)
            input_tensor = videos_tensor.to(model_device)
                   
               
            t0 = time.time()
            embeddings_tensor = g_meanClip.get_video_embeddings(input_tensor)
   
            t1 = time.time()
            print("<- g_meanClip.get_video_embeddings... ", t1 - t0, " s")

            # Convert tensor to list and add to the video_features list
            embeddings_list = embeddings_tensor.squeeze(0).tolist()

            video_features.append(embeddings_list)

        return video_features


    def load_video_for_meanclip(self, vis_path, num_frm=64, max_img_size=224, **kwargs):
        global vpath
        global vr_global
        # Load video with VideoReader
   
        if vis_path == vpath:
            pass
        else:
            t0 = time.time()
            vr_global = VideoReader(vis_path, ctx=cpu(0))
            t1 = time.time()
            vpath = vis_path
            print("**time for VideoReader**** = ", t1 - t0, " s")
        vr = vr_global
        fps = vr.get_avg_fps()
        #print("FPS",fps)
        num_frames = len(vr)

        start_idx = int(fps*kwargs.get("start_time", [0])[0])
        #print("start_idx",start_idx)
        end_idx = start_idx+int(fps*kwargs.get("clip_duration", [num_frames])[0])
        #print("end_idx",end_idx)

        frame_idx = np.linspace(start_idx, end_idx, num=num_frm, endpoint=False, dtype=int) # Uniform sampling

        clip_images = []

        # preprocess images
        clip_preprocess = get_transforms("clip_preproc", max_img_size)
        
        
        t0 = time.time()
        temp_frms = vr.get_batch(frame_idx.astype(int).tolist())
      
        t1 = time.time()
        print("time for vr.get_batch = ", t1 - t0, " s")
        t0 = time.time()
        for idx in range(temp_frms.shape[0]):
            
            im = temp_frms[idx] # H W C
            np_image = transform(np.array(im), max_img_size)
            clip_images.append(clip_preprocess(toPIL(np_image.permute(2,0,1)))) # 3, 224, 224  as input to append
            #clip_images.append(clip_preprocess(toPIL(im.permute(2,0,1)))) 
        t1 = time.time()
        print("time for preproc = ", t1 - t0, " s")
        clip_images_tensor = torch.zeros((num_frm,) + clip_images[0].shape)
        clip_images_tensor[:num_frm] = torch.stack(clip_images)
        return clip_images_tensor


class VideoVS:
    def __init__(self, host, port, selected_db, video_retriever_model,ov_device, chosen_video_search_type="similarity"):
        self.host = host
        self.port = port
        self.selected_db = selected_db
        self.chosen_video_search_type = chosen_video_search_type
        self.constraints = None
        self.video_collection = 'video-test'
        self.video_embedder = MeanCLIPEmbeddings(model=video_retriever_model,ov_device=ov_device)
        self.chosen_video_search_type = chosen_video_search_type

        # initialize_db
        self.get_db_client()
        self.init_db()


    def get_db_client(self):

        if self.selected_db == 'vdms':
            print ('Connecting to VDMS db server . . .')
            self.client = VDMS_Client(host="localhost", port=55555) #(host=self.host, port=self.port)

    def init_db(self):
        print ('Loading db instances')
        if self.selected_db == 'vdms':
            self.video_db = VDMS(
                client=self.client,
                embedding=self.video_embedder,
                collection_name=self.video_collection,
                engine="FaissFlat",
                distance_strategy="IP"
            )


    def update_db(self, prompt, n_images):
        print ('----------------------Update DB------------------------------')

        self.update_image_retriever = self.video_db.as_retriever(search_type=self.chosen_video_search_type, search_kwargs={'k':n_images})
    
    def MultiModalRetrieval(self, query: str, top_k: Optional[int] = 3):
        self.update_db(query, top_k)
        #print("After update_db")
        #video_results = self.video_retriever.invoke(query)
        video_results = self.video_db.similarity_search_with_score(query=query, k=top_k, filter=self.constraints)
        #for r, score in video_results:
        #    print("videos:", r.metadata['video_path'], '\t', r.metadata['date'], '\t', r.metadata['time'], r.metadata['timestamp'], f"score: {score}", r, '\n')

        return video_results
