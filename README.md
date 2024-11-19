# VLC Video RAG
Extension in lua for VLC RAG 

There are two parts to setting up this demo. First is the setup of docker and python on Windows Subsystem for Linux. The second part is the setup of the VLC extension on Windows.

## PART ONE: Visual RAG Setup on Window Subsystem for Linux (WSL)
In this part, you will setup WSL, install docker, and run VDMS in docker container.


### A. Setup WSL for model server
https://learn.microsoft.com/en-us/windows/wsl/install

1. You can now install everything you need to run WSL with a single command. Open PowerShell or Windows Command Prompt in administrator mode by right-clicking and selecting "Run as administrator", enter the wsl --install command, then restart your machine.
```
wsl --install
```
2. Set up your Linux user info

### B. Install Docker
https://docs.docker.com/desktop/install/ubuntu/

1. Set up Docker's apt repository
```
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
```
2.
```
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

3. Verify that the Docker Engine installation is successful by running the hello-world image.
```
sudo docker run hello-world
```
4. If you are behind a proxy make sure to configure docker for proxies:
 ```
 sudo mkdir -p /etc/systemd/system/docker.service.d
 sudo nano /etc/systemd/system/docker.service.d/http-proxy.conf
 ```
 Add the following configuration, replacing the proxy URL and port with your actual proxy details:
```
[Service]
Environment="HTTP_PROXY=http://your.proxy.address:port"
Environment="HTTPS_PROXY=http://your.proxy.address:port"
Environment="NO_PROXY=localhost,127.0.0.1"
```
Reload Systemd and Restart Docker
```
sudo systemctl daemon-reload
sudo systemctl restart docker
```

### C. Start VDMS docker container (on WSL)
1. start VDMS DB as docker container. The folder "vdms_db" will store the database on the host system to enable persistent memory : 
  ```
  sudo docker run -d -p 55555:55555 --rm --name vdms-rag -v ./vdms_db:/vdms/build/db intellabs/vdms:latest
  ```

### D. Set up visual rag repo (Windows OS)
1. Git clone the repo 
	```
	git clone <repo-name>
	```

2. create virtual env in the home directory:
   If python is not yet installed on the system, install python 3.12
   ```
   python -m venv rag_env
   ```
3. Activate the virtual env and install the requirements:
   ```
   rag_env\Scripts\activate
   cd video_retrieval
   pip install -r docs/requirements.txt
   ```
4. Install landchain-ai
   ```
   git clone https://github.com/langchain-ai/langchain
   pip install -e langchain/libs/community
   ```
5. ```
   pip install onnx==1.16.0 openvino==2024.3.0 openvino-dev==2024.3.0
   ```


## Part TWO: Run VLC Video RAG (Windows OS)
### A. Setup vlc rag lua extension 
1. copy `video_rag.lua` and `client_Rag` to `C:\Program Files\VideoLAN\VLC\lua\extensions`. 
2. After installation in a new terminal run 
	```
   python -m pip install pywin32
   ```

### B. Start video RAG 
1. Go to the command window where you have rag_env activated <br>
	`cd video_retrieval`
	
2. Update videos path in docs/config.yaml to point to the folder where you have the videos to perform Rag on. 

3. Start VLC Video Rag server. 
   - For the very first time we will need to generate and store embeddings of the videos in VDMS. Use "-g generate" flag. This step will take few minutes depending on the number of videos. Also, if it doesn't find the OpenVINO clip IRs then wait for it to convert from pytorch to onnx to openvino. Currently NPU compilation takes 29 min (This is a one time step) <br>

    ```
    python VLC_video-retrieval-ui.py -c docs/config.yaml -g generate 
    ``` 
   - For any subsequent runs, we just need to start the server as we already have the embeddings in the DB. <br>
    ```
    python VLC_video-retrieval-ui.py -c docs/config.yaml
    ```

	
4. Open VLC

5. Go to View -> Video Retrieval

6. Two ways to use vlc rag extension: <br>
   A. Pass the video folder and retrieve a video clip based on the prompt: <br>
    - Type a prompt and click on "Search" <br>
	examples: <br>
	Kayaking near a sail boat <br>
	Reef Sharks <br>
	Safari <br>
	Dog riding in a car <br>
	Dog going for a walk <br>
	Dog and Fish
	
	
	
   B. Seek to a different position in playing video based on the prompt <br>
    - Open gopro00191.mov <br>
	- Hit Pause <br>
	- search for "Sea Lion"

# Acknowledgements
Video embeddings - https://github.com/ttrigui/GenAIExamples/tree/client_inference


# License
There are two licenses associated with this project. 
The VLC lua sample code is under MIT and everything else is under Apache-2.0


# Disclaimer
Ethical Guidelines for GenerativeAI: The Responsible AI (RAI) guidance for internal use of generative AI is to follow the Intel Generative AI Guidelines as they continue to evolve, reference the  Ethics Guidance for Generative AI 1.0.docx that the RAI Council put together, and connect with the Generative AI Community of Practice to share BKMs as we learn more about this emerging technology. Please consult this guidance to develop guidelines for your participants.

# Human Rights Disclaimer
Intel is committed to respecting human rights and avoiding causing or directly contributing to adverse impacts on human rights. See Intel’s Global Human Rights Policy. See [Intel’s Global Human Rights Policy](https://www.intel.com/content/www/us/en/policy/policy-human-rights.html). The software or model from Intel is intended for socially responsible applications and should not be used to cause or contribute to a violation of internationally recognized human rights
  





