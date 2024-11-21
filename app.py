from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModel
import torch

# Flaskアプリの初期化
app = Flask(__name__)

# モデルとトークナイザーの名前
# MODEL_NAME = "cl-tohoku/bert-base-japanese"
MODEL_NAME = "line-corporation/line-distilbert-base-japanese"

# モデルとトークナイザーのロード
try:
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)  # クラス数を指定（例: 2クラス分類）
    model.eval()  # 推論モードに設定
except Exception as e:
    print(f"モデルのロード中にエラーが発生しました: {e}")

# 推論用のエンドポイント
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # リクエストからテキストデータを取得
        data = request.json
        text = data.get("text", "")
        
        if not text:
            return jsonify({"error": "テキストが空です"}), 400
        
        # トークナイズとモデル推論
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        predicted_class = logits.argmax(dim=-1).item()
        
        # 結果をJSON形式で返す
        return jsonify({"text": text, "predicted_class": predicted_class})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# メインスクリプト
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
