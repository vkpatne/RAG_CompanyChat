Step 1: Install prerequisites

1. Python 3.10+
👉 Download here : https://www.python.org/downloads/windows/

During install, check "Add Python to PATH".

2. Git for Windows
👉 Download here : https://git-scm.com/download/win

3. Git LFS
👉 Download here: https://git-lfs.com/?utm_source=chatgpt.com

After install, open PowerShell and run below command:

git lfs install

4. Hugging Face CLI, run below in powershell.

pip install -U "huggingface_hub[cli]"



🔹 Step 2: Get a Hugging Face token

C1. reate a free account: https://huggingface.co

2. Go to Settings → Access Tokens → Create a token with read role.

3. In PowerShell, login:

huggingface-cli login


Paste your token (hf_...).



🔹 Step 3: Download Mistral 7B (quantized GGUF)
I tried with below configuration. Please change model according to your configuration.
Since your CPU has 16 GB RAM, use a Q4_K_M quantization (~4.5 GB).
Run below command in poershell.

huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.1-GGUF mistral-7b-instruct-v0.1.Q4_K_M.gguf --local-dir C:\models\mistral-7b


Verify:

dir C:\models\mistral-7b


You should see:

mistral-7b-instruct-v0.1.Q4_K_M.gguf



🔹 Step 4: Install llama.cpp (console mode)

Go to llama.cpp Releases https://github.com/ggerganov/llama.cpp/releases/latest

Download the latest Windows x64 zip (e.g., llama-b6673-bin-win-cpu-x64.zip).

Extract it to C:\llama.

Inside C:\llama you should see rpc-server.exe.


✅ Test run:
To run in cli
cd C:\llama
.\llama-cli.exe -m "C:\models\mistral-7b\mistral-7b-instruct-v0.1.Q4_K_M.gguf" -p "Hello, Mistral!"

To run in browser:
 .\llama-server.exe -m "C:\models\mistral-7b\mistral-7b-instruct-v0.1.Q4_K_M.gguf"
 http://localhost:8080/


🔹 Step 5: Install Text Generation WebUI (GUI) (Optional)

Clone repo:

git clone https://github.com/oobabooga/text-generation-webui.git C:\textgen


Install requirements:

cd C:\textgen
pip install -r requirements.txt


Run the WebUI with llama.cpp backend:

python server.py --model "C:\models\mistral-7b" --loader llama.cpp


Open browser at:
👉 http://localhost:7860



🔹 Step 6: (Optional) Set environment variable

If you want programs to always know where the model is:

setx MISTRAL_MODEL_PATH "C:\models\mistral-7b"



✅ Summary

Console use:

cd C:\llama
.\main.exe -m "C:\models\mistral-7b\mistral-7b-instruct-v0.1.Q4_K_M.gguf" -p "Write a haiku about autumn"


GUI use:

cd C:\textgen
python server.py --model "C:\models\mistral-7b" --loader llama.cpp


→ open http://localhost:7860

⚡ Now you can run Mistral both from command line and with a browser chat interface.
