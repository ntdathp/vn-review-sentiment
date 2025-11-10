# Environment
For Phobert python version 3.10.18

    git clone git@github.com:ntdathp/nlp_learning.git
    cd nlp_learning

<details>
  <summary> For virtual environment. Click here to expand</summary>

    # Install pyenv + build deps (Ubuntu/Debian)
    sudo apt update
    sudo apt install -y build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
    libffi-dev liblzma-dev
    curl https://pyenv.run | bash
    
    # Add pyenv to shell
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    exec $SHELL  # hoặc: source ~/.bashrc
    
    # Install Python 3.10.18
    pyenv install 3.10.18
    pyenv local 3.10.18     
    python -V                # check Python 3.10.18

    # Create and activate venv
    python -m venv .venv310
    source .venv310/bin/activate
    python -V  # 3.10.18
    pip install --upgrade pip setuptools whee
    
</details>


```
pip install -r requirements.txt
```
# Test Phobert

Change the path to trained model in my_phobert_only_cpu.py or my_phobert_only.py. To run on cpu or gpu respectively.
Download trained model [here](https://drive.google.com/drive/folders/1VI1AyaTUFOaKDCyZbS8mfxUguh9thrt9?usp=sharing). Remember to extract zip file.

```python
_DEF_CANDIDATES: List[str] = [
    os.environ.get("PHOBERT_MODEL_DIR", ""),                
    "/home/dat/llm_ws/phobert/phobert_5cls_clean",           # path 
]
```
Then choose my_phobert_only_cpu or my_phobert_only in chat_toolbox.py.

```python
mod_name = os.environ.get("CHAT_TOOLBOX_PHOBERT_MODULE", "my_phobert_only_cpu")
```

Finally run the GUI.
```
cd pip phobert
python chat_toolbox.py
```

# Test LLM
Read 



