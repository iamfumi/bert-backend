from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS  # CORSをインポート
from bs4 import BeautifulSoup
import urllib.request
import datetime
import MeCab
import re
import csv
from json import dumps

app = Flask(__name__)
CORS(app)  # CORSを適用

import MeCab

@app.route('/getScoreMecab', methods=['POST'])
def get_score_mecab():
    # 辞書パスを指定
    mecab_args = '-r /etc/mecabrc -d /usr/lib/x86_64-linux-gnu/mecab/dic/ipadic'
    try:
        m = MeCab.Tagger(mecab_args)
        result = m.parse("テスト")
        return result
    except RuntimeError as e:
        print("MeCab initialization failed:", str(e))
        return {"error": "MeCab initialization failed"}, 500

# 静的ファイルの提供
@app.route('/file/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

# ログインページ（GETリクエスト）
@app.route('/login', methods=['GET'])
def login():
    return '''
        <form action="/getScore" method="post">
            <p>Your Review:<br>
            <textarea name="rev" rows="4" cols="40"></textarea>
            </p>
            <p>
            <input type="submit" value="送信"><input type="reset" value="リセット">
            </p>
        </form>
    '''

# スコアを計算するエンドポイント（POSTリクエスト）
@app.route('/getScorelotion', methods=['POST'])
def get_score_lotion():
    lotionDict = []
    testSentence = request.form['rev']

    # 辞書ファイルの読み込み
    with open('./lotiondic0712.csv', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            inputArray = [row[0], row[1], row[2], row[3], row[4], row[5]]
            lotionDict.append(inputArray)

    # MeCabを使った形態素解析
    m = MeCab.Tagger()
    s1 = m.parse(testSentence)
    
    # 文を区切る
    pattern = re.compile('、|。|！|\!|\)|\?|？|♪|★|☆')
    s = pattern.split(s1)

    Tscore = {i: 0 for i in range(10)}
    Tnum = {i: 0 for i in range(10)}

    # テキスト分析
    for line in s:
        words = line
        inputLine = line
        for eachEntry in lotionDict:
            if words.find(eachEntry[0]) != -1:
                tempCheckArray = [int(bool(eachEntry[i])) for i in range(1, 4)]
                flagAll = all(inputLine.find(eachEntry[i]) != -1 if tempCheckArray[i-1] else True for i in range(1, 4))
                if flagAll:
                    sc = float(eachEntry[4])
                    ca = int(eachEntry[5])
                    if 'ない' not in line or eachEntry[3]:
                        Tscore[ca] += sc
                        Tnum[ca] += 1

    # スコアとタグを計算
    evScoreArray = [round(Tscore[i] / Tnum[i]) if Tnum[i] != 0 else 0 for i in range(10)]
    tagArray = ["香り", "うるおい/浸透", "美白/UV", "毛穴ケア", "爽快感", "サラサラ", "低刺激", "肌荒れ対策", "エイジングケア", "コスパ"]

    # レスポンスデータの作成
    response = {
        "result": "ok",
        "data": {
            "evScore": evScoreArray,
            "tags": [tagArray[i] if evScoreArray[i] > 3 else "" for i in range(10)]
        }
    }

    return jsonify(response)

# スコアを計算するエンドポイント（POSTリクエスト）
@app.route('/getScore', methods=['POST'])
def get_score():
    shampooDict = []
    testSentence = request.form['rev']

    # 辞書ファイルの読み込み
    with open('./shampoodic_floral_15.csv', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            inputArray = [row[0], row[1], row[2], row[3], row[4], row[5]]
            shampooDict.append(inputArray)

    # MeCabを使った形態素解析
    m = MeCab.Tagger()
    s1 = m.parse(testSentence)
    
    # 文を区切る
    pattern = re.compile('、|。|！|\!|\)|\?|？|♪|★|☆')
    s = pattern.split(s1)

    Tscore = {i: 0 for i in range(10)}
    Tnum = {i: 0 for i in range(10)}

    # テキスト分析
    # for line in s:
    #     words = line
    #     inputLine = line
    #     for eachEntry in shampooDict:
    #         if words.find(eachEntry[0]) != -1:
    #             tempCheckArray = [int(bool(eachEntry[i])) for i in range(1, 4)]
    #             flagAll = all(inputLine.find(eachEntry[i]) != -1 if tempCheckArray[i-1] else True for i in range(1, 4))
    #             if flagAll:
    #                 sc = float(eachEntry[4])
    #                 ca = int(eachEntry[5])
    #                 if 'ない' not in line or eachEntry[3]:
    #                     Tscore[ca] += sc
    #                     Tnum[ca] += 1
    for line in s:  # s:１文のリスト
        words = line
        inputLine = line

        for eachEntry in shampooDict:  # 辞書をまわす
            if words.find(eachEntry[0]) != -1:  # eachEntry[0](キーワード)がレビュー内に見つかったら
                tempCheckArray = [1]
                if eachEntry[1]:
                    tempCheckArray.append(1)

                else:
                    tempCheckArray.append(0)

                if eachEntry[2]:
                    tempCheckArray.append(1)

                else:
                    tempCheckArray.append(0)

                if eachEntry[3]:
                    tempCheckArray.append(1)

                else:
                    tempCheckArray.append(0)

                flagAll = 1
                counter = 0

                for eachCheck in tempCheckArray:
                    if tempCheckArray[counter] == 1:
                        if inputLine.find(eachEntry[counter]) == -1:
                            flagAll = 0

                    counter = counter + 1

                if flagAll == 1:
                    sc = float(eachEntry[4])  # 文字列のeachEntry[4](スコア)を数値に変換
                    ca = int(eachEntry[5])    # 文字列のeachEntry[5](評価項目)を数値に変換

                    if not ('ない' in line) or eachEntry[3]:
                        Tscore[ca] = sc
                        Tnum[ca] = 1

    # スコアとタグを計算
    # evScoreArray = [round(Tscore[i] / Tnum[i]) if Tnum[i] != 0 else 0 for i in range(10)]
    evScoreArray = [round(Tscore[i] / 1) if Tnum[i] != 0 else 0 for i in range(10)]
    tagArray = ["フローラル", "うるおい/浸透", "美白/UV", "毛穴ケア", "爽快感", "サラサラ", "低刺激", "肌荒れ対策", "エイジングケア", "コスパ"]

    # レスポンスデータの作成
    response = {
        "result": "ok",
        "data": {
            "evScore": evScoreArray,
            "tags": [tagArray[i] if evScoreArray[i] > 3 else "" for i in range(10)]
        }
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
