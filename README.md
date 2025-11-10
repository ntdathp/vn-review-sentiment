# Virtual environment
For Phobert python version 3.10.18

```
pip install -r requirements.txt
```
# Test Phobert

Change the path to trained model in my_phobert_only_cpu.py or my_phobert_only.py. To run on cpu or gpu respectively.
Download trained model [here](https://drive.google.com/drive/folders/1VI1AyaTUFOaKDCyZbS8mfxUguh9thrt9?usp=sharing).

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



