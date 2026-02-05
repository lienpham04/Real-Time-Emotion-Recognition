# File: server/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import numpy as np
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import emoji
from transformers import AutoTokenizer, AutoModel
from deep_translator import GoogleTranslator 

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Server is running on: {DEVICE}")

BACKBONE = "microsoft/deberta-v3-large"
MAX_LEN = 128
HEF_DIM = 13
NUM_LABELS = 28
EMOTION_NAMES = [
 'admiration','amusement','anger','annoyance','approval','caring','confusion','curiosity','desire',
 'disappointment','disapproval','disgust','embarrassment','excitement','fear','gratitude','grief',
 'joy','love','nervousness','optimism','pride','realization','relief','remorse','sadness','surprise','neutral'
]

# Negative emotions list (logic VADER)
NEGATIVE_EMOTIONS = [
    'anger', 'annoyance', 'disappointment', 'disgust', 'fear', 
    'grief', 'remorse', 'sadness', 'nervousness', 'embarrassment'
]

# Thresholds for each emotion
THRESHOLDS = [
    0.40, # admiration
    0.55, # amusement
    0.35, # anger
    0.30, # annoyance
    0.30, # approval
    0.30, # caring
    0.30, # confusion
    0.30, # curiosity
    0.25, # desire
    0.25, # disappointment
    0.30, # disapproval
    0.30, # disgust
    0.20, # embarrassment
    0.35, # excitement
    0.30, # fear
    0.60, # gratitude
    0.15, # grief
    0.45, # joy
    0.40, # love
    0.20, # nervousness
    0.35, # optimism
    0.30, # pride
    0.25, # realization
    0.15, # relief
    0.20, # remorse
    0.35, # sadness
    0.35, # surprise
    0.3  # neutral
]


MODEL_FILE_NAME = "best_model.pt"

# ---------------------------------------------------------
# 2. NLP TOOLS INITIALIZATION
# ---------------------------------------------------------
print("Initializing NLP tools...")
nlp = spacy.load("en_core_web_sm", disable=["ner"])
vader = SentimentIntensityAnalyzer()
# use_fast=False để tránh lỗi protobuf
tokenizer = AutoTokenizer.from_pretrained(BACKBONE, use_fast=False) 
translator = GoogleTranslator(source='auto', target='en')

# ---------------------------------------------------------
# 3. DEFINE MODEL & HEF
# ---------------------------------------------------------
def extract_hef_features(text):
    doc = nlp(text)
    pos_counts = {'ADJ':0,'ADV':0,'INTJ':0,'VERB':0,'NOUN':0}
    for tok in doc:
        if tok.pos_ in pos_counts:
            pos_counts[tok.pos_] += 1
    exclam = text.count('!')
    qmarks = text.count('?')
    allcaps = sum(1 for w in text.split() if w.isupper() and len(w) > 1)
    vs = vader.polarity_scores(text)
    emojis = sum(1 for c in text if c in emoji.EMOJI_DATA)
    feat = [
        pos_counts['ADJ'], pos_counts['ADV'], pos_counts['INTJ'],
        pos_counts['VERB'], pos_counts['NOUN'],
        exclam, qmarks, allcaps,
        vs['neg'], vs['neu'], vs['pos'], vs['compound'],
        emojis
    ]
    return np.array(feat, dtype=np.float32)

class HybridClassifier(nn.Module):
    def __init__(self, backbone, hef_dim, hidden_size, num_labels, proj=512, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(hidden_size + hef_dim, proj),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(proj, num_labels)
        )
    def forward(self, input_ids=None, attention_mask=None, hef=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = out.last_hidden_state[:,0,:]
        if hef is None:
            hef = torch.zeros((pooled.size(0), HEF_DIM), device=pooled.device)
        x = torch.cat([pooled, hef], dim=-1)
        logits = self.head(x)
        return logits

# ---------------------------------------------------------
# 4. LOAD MODEL
# ---------------------------------------------------------
print("Loading Backbone (this takes time)...")
backbone_model = AutoModel.from_pretrained(BACKBONE)
hidden_size = backbone_model.config.hidden_size
model = HybridClassifier(backbone_model, HEF_DIM, hidden_size, NUM_LABELS)

print(f"Loading Weights from {MODEL_FILE_NAME}...")
try:
    model.load_state_dict(torch.load(MODEL_FILE_NAME, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(">>> MODEL READY!")
except Exception as e:
    print(f"ERROR LOADING WEIGHTS: {e}")
    print("⚠️ LƯU Ý: Hãy chắc chắn tên file .pt trong code trùng với tên file bạn tải về.")

# ---------------------------------------------------------
# 5. API SETUP
# ---------------------------------------------------------
app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict_endpoint(req: TextRequest):
    raw_text = req.text
    if not raw_text or len(raw_text.strip()) == 0:
        return {"emotions": []}

    # 1. Dịch sang tiếng Anh
    try:
        translated_text = translator.translate(raw_text)
    except:
        translated_text = raw_text # Fallback nếu lỗi mạng

    # 2. Xử lý Input & Dự đoán (AI Model)
    enc = tokenizer(translated_text, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='pt')
    input_ids = enc['input_ids'].to(DEVICE)
    attention_mask = enc['attention_mask'].to(DEVICE)
    
    # Extract HEF trên text tiếng Anh
    hef_numpy = extract_hef_features(translated_text)
    hef_tensor = torch.from_numpy(hef_numpy).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, hef=hef_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    # 3. Tính điểm Sentiment bằng VADER (Rule-based)
    vs = vader.polarity_scores(translated_text)
    vader_neg_score = vs['neg']      
    vader_compound = vs['compound']  

    # ---------------------------------------------------------
    # 4. HYBRID LOGIC
    # ---------------------------------------------------------
    
    # Bước A: Thu thập kết quả thô từ AI
    candidates = []
    for i in range(NUM_LABELS):
        candidates.append({
            "label": EMOTION_NAMES[i],
            "score": float(probs[i]),
            "is_negative": EMOTION_NAMES[i] in NEGATIVE_EMOTIONS
        })
    
    # Sắp xếp theo điểm AI giảm dần
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    final_results = []
    
    # Bước B: Kiểm tra trường hợp VADER cứu cánh
    # NẾU (AI bảo Neutral HOẶC điểm cao nhất quá thấp) VÀ (VADER bảo Tiêu cực mạnh)
    top_prediction = candidates[0]
    
    if (top_prediction['label'] == 'neutral' or top_prediction['score'] < 0.3) and vader_compound < -0.05:
        print(f"-> Hybrid Logic: VADER detected negativity (Compound: {vader_compound}). Adjusting...")
        
        # Tìm cảm xúc TIÊU CỰC có điểm cao nhất trong danh sách AI
        best_neg_candidate = None
        for item in candidates:
            if item['is_negative']:
                best_neg_candidate = item
                break
        
        # Nếu tìm thấy, ép nó hiển thị
        if best_neg_candidate:
            final_results.append({
                "label": best_neg_candidate['label'],
                # Lấy điểm cao hơn giữa AI score và độ lớn VADER
                "score": max(best_neg_candidate['score'], abs(vader_compound)) 
            })
    
    # Bước C: Nếu không phải trường hợp cần cứu, chạy logic lọc ngưỡng bình thường
    if not final_results:
        has_emotion = False
        # Duyệt các nhãn trừ Neutral
        for item in candidates:
            if item['label'] == 'neutral': continue
            
            # Lấy ngưỡng của nhãn đó
            idx = EMOTION_NAMES.index(item['label'])
            threshold = THRESHOLDS[idx]
            
            if item['score'] >= threshold:
                final_results.append(item)
                has_emotion = True
        
        # Xử lý Neutral cuối cùng (Logic thông minh)
        neutral_item = next((x for x in candidates if x['label'] == 'neutral'), None)
        if neutral_item:
            idx = EMOTION_NAMES.index('neutral')
            # Chỉ hiện Neutral nếu không có cảm xúc nào khác VÀ điểm nó cao hơn ngưỡng của nó
            if not has_emotion and neutral_item['score'] >= THRESHOLDS[idx]:
                final_results.append(neutral_item)

    # Format lại output
    final_results.sort(key=lambda x: x['score'], reverse=True)
    
    output_emotions = [{"label": x["label"], "score": x["score"]} for x in final_results]

    return {
        "original": raw_text,
        "translated": translated_text,
        "emotions": output_emotions
    }
    
# bash: uvicorn main:app --reload