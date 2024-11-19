#Copyright (C) 2024 Intel Corporation
#SPDX-License-Identifier: Apache-2.0

import socket
import sys
import os
from pathlib import Path

# import win32gui, win32con

# hide = win32gui.GetForegroundWindow()
# win32gui.ShowWindow(hide, win32con.SW_HIDE)
user_data_dir = Path.home() / 'AppData' / 'Roaming' / 'vlc'

file_flag=os.path.join(user_data_dir,"cache_test.txt") 
try:
    #print("I am here")
    os.remove(file_flag)
except:

    None

def start_client():
    host = 'localhost'  # The server's hostname or IP address
    port = 65432 
    prompt = sys.argv[2] #"What is in the video?"

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((host, port))
        output = sys.argv[1] #'Hello Server'
        client_socket.sendall(output.encode())
        data = client_socket.recv(1024) 
 
            
            #print(f"Received from server: {data.decode()}")
        if data.decode() == "Ready":
            client_socket.sendall(prompt.encode())
            
        data = client_socket.recv(1024) 

        file_name = data.decode() 
        extensionsToCheck = ['.mp4', '.mov', '.mkv', '.avi','.MP4', '.MOV', '.MKV', '.AVI']
        if any(ext in output for ext in extensionsToCheck):
       #     frame_num = file_name.split("_")[2]
             frame_num = file_name.split(" ")[1]
        else:

             video_path = os.path.join(output,file_name)
             frame_num = "file:///" + video_path.replace("\\","/")
        
                
            
        print(frame_num)
        with open(file_flag, "w") as f:
            #print("file written")
            f.write("Got result")


        client_socket.close()


if __name__ == "__main__":

    start_client()



