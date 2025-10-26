
## 1. Setup Environment
### if you are on windows: start
### make sure you have Install Build Tools for Visual Studio → include Desktop development with C++.
Close your current terminal/command prompt.
Open the Windows Start Menu.
Open x64 Native Tools Command Prompt for VS 20XX (important: x64).
Launch this specialized command prompt.

1. To Create your virtual environment
conda create -p venv python=3.11

2. To activate virtual environment
conda activate venv/

3. To install all dependecies
pip install -r requirements.txt

#### 2. How to run
uvicorn main:app --reload


#### if running on windows create .env file at root level.
.env
EMBED_MODEL_ID="intfloat/e5-base-v2"
EMBED_LOCAL_DIR="C://models//e5_base_v2//"
LLM_LOCAL="C://models//tinyllma//tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
FORCE_OFFLINE=True
