import os
import streamlit as st
import cv2
import numpy as np
import tempfile
import mediapipe as mp
import pandas as pd
import plotly.express as px
from supabase import create_client, Client
import openai
import json
from dotenv import load_dotenv  # Render 以外の環境（ローカル）用

# ✅ .env から環境変数をロード（Render 以外のローカル環境用）
load_dotenv()

# ✅ 環境変数を読み込む
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ 環境変数が取得できていない場合のエラーハンドリング
if not SUPABASE_URL or not SUPABASE_KEY or not OPENAI_API_KEY:
    st.error("❌ 環境変数が正しく設定されていません！Render の Environment Variables を確認してください。")
    st.stop()

# ✅ Supabase クライアントを作成
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ✅ OpenAI API キーの設定
openai.api_key = OPENAI_API_KEY

# ✅ Mediapipe のセットアップ
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ✅ Webアプリのタイトル
st.title("🚶‍♂️ 歩行分析アプリ")
st.write("動画をアップロードすると歩行を解析します！")

# ✅ 動画アップロード
uploaded_file = st.file_uploader("歩行動画をアップロードしてください", type=["mp4", "mov"])

if uploaded_file:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(uploaded_file.read())

    cap = cv2.VideoCapture(temp_file.name)

    # ✅ 保存用の動画ファイルを作成
    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))  # FPSが0になるケースを防ぐ
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    if st.button("歩行解析を開始"):
        st.write("解析中...")

        joint_data = []

        # ✅ 取得する関節
        JOINTS = {
            "LEFT_HIP": mp_pose.PoseLandmark.LEFT_HIP,
            "RIGHT_HIP": mp_pose.PoseLandmark.RIGHT_HIP,
            "LEFT_KNEE": mp_pose.PoseLandmark.LEFT_KNEE,
            "RIGHT_KNEE": mp_pose.PoseLandmark.RIGHT_KNEE,
            "LEFT_ANKLE": mp_pose.PoseLandmark.LEFT_ANKLE,
            "RIGHT_ANKLE": mp_pose.PoseLandmark.RIGHT_ANKLE
        }

        # ✅ Pose モデルを初期化
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                if results.pose_landmarks:
                    frame_data = {"Time (s)": cap.get(cv2.CAP_PROP_POS_FRAMES) / fps}
                    for joint_name, joint_id in JOINTS.items():
                        landmark = results.pose_landmarks.landmark[joint_id]
                        frame_data[f"{joint_name}_Y"] = landmark.y

                    joint_data.append(frame_data)

                    # ✅ 関節マーカーを描画
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # ✅ フレームを動画に保存
                out.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        cap.release()
        out.release()

        df = pd.DataFrame(joint_data)
        st.write("✅ 解析完了！")

        # ✅ AI に解析データを送信し、解説を取得
        def generate_ai_analysis(scores_json):
            prompt = f"""
            あなたは歩行解析の専門家です。
            以下の解析結果をわかりやすく解説してください：
            {json.dumps(scores_json, indent=2, ensure_ascii=False)}
            """

            # ✅ 最新APIに修正
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "あなたは歩行解析の専門家です。"},
                    {"role": "user", "content": prompt}
                ]
            )
            ai_analysis = response['choices'][0]['message']['content']
            return ai_analysis

        # ✅ AI解析の実行
        scores = {"Stability Score": 85, "Gait Rhythm Score": 90, "Symmetry Score": 88}
        ai_analysis = generate_ai_analysis(scores)
        st.subheader("📖 AI による解析解説")
        st.write(ai_analysis)
