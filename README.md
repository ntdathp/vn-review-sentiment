# nlp_learning
Tớ học lại NLP

# Virtual environment

3.8 for bilstm

source ~/llm_ws/.venv/bin/activate

3.10 for Phobert

version = 3.10.18

source ~/llm_ws/.venv310/bin/activate

muốn test các code chạy môi trường python 3.10 thì cài pip các gói ở file requirement.txt

# Test bilstm
python bilstm/predict_one.py --model_dir bilstm_vn_sentiment_multiclass \
  --text "Thiết bị robot hút bụi khien toi thất vọng ồn shop phản hồi chậm."

# Test Phobert
python3 phobert/test_phobert.py --model_dir /home/dat/llm_ws/phobert_5cls_clean   --text "Thiết bị robot hút bụi thất vọng ồn shop phản hồi chậm."

# Test LLM
python3 llm/classify_csv_llm.py "sản phẩm laptop rất kinh khủng, gia công cực kinh khủng; phản hồi chậm, rất bực mình."

ollama ps

ollama stop qwen2.5:14b-instruct


