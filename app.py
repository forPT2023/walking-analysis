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

# ✅ 環境変数を読み込む
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ 環境変数が取得できていない場合のエラーハンドリング
if not SUPABASE_URL or not SUPABASE_KEY or not OPENAI_API_KEY:
    st.error("❌ 環境変数が正しく設定されていません！Render または Streamlit Cloud の Environment Variables を確認してください。")
    st.stop()

# ✅ Supabase クライアントを作成
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ✅ OpenAI API キーの設定
openai.api_key = OPENAI_API_KEY

# ✅ Mediapipe のセットアップ
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ✅ ページ設定でモバイル表示を最適化
st.set_page_config(
    page_title="歩行分析アプリ",
    page_icon="🚶‍♂️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ✅ カスタムCSS
st.markdown("""
    <style>
    .stApp {
        max-width: 100%;
        padding: 1rem;
    }
    .uploadedFile {
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ✅ Webアプリのタイトル
st.title("🚶‍♂️ 歩行分析アプリ")

# ✅ 説明文をexpanderに格納してスペースを節約
with st.expander("📱 使い方を見る"):
    st.write("""
    1. 歩行動画をアップロードします
    2. 「歩行解析を開始」ボタンをタップします
    3. 解析結果が表示されるまでお待ちください
    """)

# ✅ 動画アップロード
uploaded_file = st.file_uploader("歩行動画をアップロード 📸", type=["mp4", "mov"])

if uploaded_file:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(uploaded_file.read())

    cap = cv2.VideoCapture(temp_file.name)

    # ✅ 保存用の動画ファイルを作成
    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    if st.button("👉 歩行解析を開始"):
        with st.spinner("🔄 解析中..."):
            joint_data = []

            JOINTS = {
                "LEFT_HIP": mp_pose.PoseLandmark.LEFT_HIP,
                "RIGHT_HIP": mp_pose.PoseLandmark.RIGHT_HIP,
                "LEFT_KNEE": mp_pose.PoseLandmark.LEFT_KNEE,
                "RIGHT_KNEE": mp_pose.PoseLandmark.RIGHT_KNEE,
                "LEFT_ANKLE": mp_pose.PoseLandmark.LEFT_ANKLE,
                "RIGHT_ANKLE": mp_pose.PoseLandmark.RIGHT_ANKLE
            }

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
                        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    out.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            cap.release()
            out.release()

            df = pd.DataFrame(joint_data)
            st.success("✅ 解析完了！")

            # ✅ タブで結果を整理
            tab1, tab2, tab3 = st.tabs(["📊 グラフ", "🎥 動画", "📝 解説"])
            
            with tab1:
                # ✅ グラフを最適化
                fig = px.line(df, x="Time (s)", 
                            y=["LEFT_KNEE_Y", "RIGHT_KNEE_Y", "LEFT_ANKLE_Y", "RIGHT_ANKLE_Y"],
                            title="歩行バランスの変化",
                            labels={"value": "関節の高さ", "variable": "関節"})
                fig.update_layout(
                    height=400,
                    margin=dict(l=10, r=10, t=30, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.video(output_video_path)
                with open(output_video_path, "rb") as file:
                    st.download_button(
                        "📥 解析動画をダウンロード",
                        file,
                        file_name="walking_analysis.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )

            with tab3:
                # ✅ スコア計算
                def calculate_gait_scores(df):
                    scores = {}
                    scores["Stability Score"] = max(0, 100 - (df["LEFT_KNEE_Y"].std() + df["RIGHT_KNEE_Y"].std()) * 50)
                    step_intervals = np.diff(df["Time (s)"])
                    scores["Gait Rhythm Score"] = max(0, 100 - np.std(step_intervals) * 500)
                    scores["Symmetry Score"] = max(0, 100 - np.mean(np.abs(df["LEFT_KNEE_Y"] - df["RIGHT_KNEE_Y"])) * 500)
                    return scores

                scores = calculate_gait_scores(df)
                
                # ✅ スコアをカード形式で表示
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("安定度", f"{scores['Stability Score']:.1f}")
                with col2:
                    st.metric("リズム", f"{scores['Gait Rhythm Score']:.1f}")
                with col3:
                    st.metric("対称性", f"{scores['Symmetry Score']:.1f}")

                # ✅ AI解析
                def generate_ai_analysis(scores_json):
                    prompt = f"""
                    あなたは歩行解析の専門家です。
                    以下の解析結果を簡潔にわかりやすく解説してください：
                    {json.dumps(scores_json, indent=2, ensure_ascii=False)}
                    """

                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "あなたは歩行解析の専門家です。"},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    return response["choices"][0]["message"]["content"]

                with st.spinner("AI解析中..."):
                    ai_analysis = generate_ai_analysis(scores)
                    st.write(ai_analysis)