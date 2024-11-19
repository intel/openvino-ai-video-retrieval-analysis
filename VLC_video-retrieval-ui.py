#Copyright (C) 2024 Intel Corporation
#SPDX-License-Identifier: Apache-2.0

import os
from embedding.vector_stores import db
import time

import socket
import argparse

from utils import config_reader as reader
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

from embedding.generate_store_embeddings import setup_meanclip_model 

import decord
decord.bridge.set_bridge('torch')


# Create argument parser
parser = argparse.ArgumentParser(description='Process configuration file for generating and storing embeddings.')

parser.add_argument('-c','--config_file', type=str, help='Path to configuration file (e.g., config.yaml)',required=True)
parser.add_argument('-g', '--generate', type=str, help='Flag to enable generate & store embedding in DB',required=False, default=None)

# Parse command-line arguments
args = parser.parse_args()
# Read configuration file

if args.config_file.endswith('.yaml'):
    config = reader.read_config(args.config_file)
else:
    print("Config file should be yaml format")
    import sys
    sys.exit(1)

device = "cpu"
video_dir = config['videos']
# Read MeanCLIP

if args.generate == "generate":
    print("---------GENERATE AND STORE EMBEDDINGS-----------")
    metadata_file = os.path.join(config['meta_output_dir'], "metadata.json")
    if os.path.exists(metadata_file):
        os.remove(metadata_file)

#if not os.path.exists(os.path.join(config['meta_output_dir'], "metadata.json")):
    from embedding.generate_store_embeddings import main
    vs = main(config)
    #Update descriptors for backup
    import vdms

    db_vdms = vdms.vdms()
    DBHOST=config['vector_db']['host']
    DBPORT = int(config['vector_db']['port'])

    db_vdms.connect(DBHOST, DBPORT)

    response, _ = db_vdms.query([{"FindDescriptorSet": {"set":'video-test',"storeIndex": True}}])
  

session_state = {} 

    
def get_top_doc(results, qcnt):
    if qcnt < len(results):
        print("video retrieval done")
        return results[qcnt]
    return None


        
if 'vs' not in session_state.keys():
        print('Preparing RAG pipeline')
        time.sleep(1)
        host = config['vector_db']['host']
        port = int(config['vector_db']['port'])
        selected_db = config['vector_db']['choice_of_db']
        try:
            
            session_state['vs'] = vs
        except:
         
            if config['embeddings']['type'] == "frame":
                session_state['vs'] = db.VS(host, port, selected_db)
            elif config['embeddings']['type'] == "video":
                import json
                meanclip_cfg_json = json.load(open(config['meanclip_cfg_path'], 'r'))
                meanclip_cfg = argparse.Namespace(**meanclip_cfg_json)
                model, _ = setup_meanclip_model(meanclip_cfg, device="cpu")
                session_state['vs'] = db.VideoVS(host, port, selected_db, model,config['embeddings']['device']) 

    

def RAG(prompt):
    

    results = session_state['vs'].MultiModalRetrieval(prompt, top_k = 3) #n_texts = 1, n_images = 3)
        #status.update(label="Retrieved top matching clip!", state="complete", expanded=False)
  
    print (f'\tRAG prompt={prompt}')
   
      
    result = get_top_doc(results, session_state["qcnt"])
    if result == None:
        return None
    try:
        top_doc, score = result
    except:
        top_doc = result
    print('TOP DOC = ', top_doc.metadata['video'])
    print("PLAYBACK OFFSET = ", top_doc.metadata['timestamp'])
    
    return top_doc


if 'prevprompt' not in session_state.keys():
    session_state['prevprompt'] = ''
    print("Setting prevprompt to None")
if 'prompt' not in session_state.keys():
    session_state['prompt'] = ''
if 'qcnt' not in session_state.keys():
    session_state['qcnt'] = 0

def handle_message(prompt):
    import math
    print("-"*30)
    print("starting message handling")
    # Generate a new response if last message is not from assistant


    top_doc = RAG(prompt)

    if top_doc == None:
        
        print("No more relevant videos found. Select a different query. or RAG didnt work check your DB")
        #placeholder.markdown(full_response)

    else:
        video_name, playback_offset, video_path = top_doc.metadata['video'], float(top_doc.metadata['timestamp']), top_doc.metadata['video_path']

        videoname = os.path.basename(video_name).split('_interval')[0] + ".mp4" + " " + str(playback_offset)
        print("Final video name",videoname)
        return videoname


print("Starting RAG")

host = 'localhost'
port = 65432

#handle_message()
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((host, port))
    s.listen()
    print(f"Listening on {host}:{port}")
    while True:
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            while True:
                print("Waiting")
        
                data = conn.recv(1024)
                file_path = data.decode()
                
                if not data:
                    break
                

                print("video path recieved", file_path)
                #extensionsToCheck = ['.mp4', '.mov', '.mkv', '.avi']

                #if any(ext in file_path for ext in extensionsToCheck):
                conn.sendall(b'Ready')
                data = conn.recv(1024)
                prompt = data.decode()
                print(f"Received from client: {prompt}")  
                outputs = handle_message(prompt)
                if outputs:
                    conn.sendall(outputs.encode())
                else:
                    out="RAG no working"
                    conn.sendall(out.encode())



