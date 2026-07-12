import streamlit as st
import os
import cv2
import uuid
import tempfile
import numpy as np
import time
from collections import Counter
from deepface import DeepFace

# ===== 语音情感分析依赖 =====
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
# ================================

# ---------- 配置 ----------
KNOWN_FACES_DIR = "known_faces"
DISTANCE_THRESHOLD = 0.4
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "yolov11n"
AUTO_STORE_UNKNOWN = True

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

if "streaming" not in st.session_state:
    st.session_state.streaming = False
if "stop_stream" not in st.session_state:
    st.session_state.stop_stream = False
if "fusion_result" not in st.session_state:
    st.session_state.fusion_result = None

# ===== 加载 SenseVoiceSmall 模型 =====
@st.cache_resource
def load_sensevoice_model():
    try:
        model = AutoModel(
            model="iic/SenseVoiceSmall",
            trust_remote_code=True,
            device="cpu",
        )
        return model
    except Exception as e:
        st.error(f"加载语音模型失败: {e}")
        return None

# ---------- 辅助函数 ----------
def store_unknown_faces(img_path: str) -> list:
    try:
        face_objs = DeepFace.extract_faces(
            img_path=img_path,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True
        )
        if not face_objs:
            return []
        saved_paths = []
        for i, face_obj in enumerate(face_objs):
            face_img = face_obj["face"]
            face_img_bgr = (face_img * 255).astype("uint8")
            face_img_bgr = cv2.cvtColor(face_img_bgr, cv2.COLOR_RGB2BGR)
            filename = f"unknown_{uuid.uuid4().hex}_{i}.jpg"
            save_path = os.path.join(KNOWN_FACES_DIR, filename)
            cv2.imwrite(save_path, face_img_bgr)
            saved_paths.append(save_path)
        return saved_paths
    except Exception as e:
        st.error(f"存储人脸失败: {e}")
        return []

def process_multi_faces(img_path: str, auto_store: bool = AUTO_STORE_UNKNOWN):
    try:
        face_objs = DeepFace.extract_faces(
            img_path=img_path,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True
        )
    except Exception as e:
        st.error(f"人脸检测失败: {e}")
        return [], [], []

    if not face_objs:
        st.warning("未检测到任何人脸")
        return [], [], []

    face_images = [obj["face"] for obj in face_objs]
    display_faces = []
    for face in face_images:
        face_uint8 = (face * 255).astype(np.uint8)
        display_faces.append(face_uint8)

    attribute_results = []
    try:
        analysis_list = DeepFace.analyze(
            img_path=img_path,
            actions=['emotion'],
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
            silent=True
        )
        for a in analysis_list:
            attribute_results.append({'emotion': a.get('dominant_emotion')})
    except Exception as e:
        st.error(f"属性分析失败: {e}")
        attribute_results = [None] * len(face_objs)

    identity_results = []
    if not os.listdir(KNOWN_FACES_DIR):
        for i in range(len(face_objs)):
            identity_results.append({'is_familiar': False, 'name': None, 'distance': None, 'stored_path': None})
        if auto_store:
            stored_paths = store_unknown_faces(img_path)
            for i, sp in enumerate(stored_paths):
                if i < len(identity_results):
                    identity_results[i]['stored_path'] = sp
    else:
        try:
            find_results = DeepFace.find(
                img_path=img_path,
                db_path=KNOWN_FACES_DIR,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                distance_metric="cosine",
                enforce_detection=True,
                silent=True
            )
            for i, df in enumerate(find_results):
                if df is not None and not df.empty:
                    best = df.iloc[0]
                    distance = best['distance']
                    best_path = best['identity']
                    name = os.path.splitext(os.path.basename(best_path))[0]
                    is_familiar = distance < DISTANCE_THRESHOLD
                    identity_results.append({'is_familiar': is_familiar, 'name': name if is_familiar else None, 'distance': distance, 'stored_path': None})
                else:
                    identity_results.append({'is_familiar': False, 'name': None, 'distance': None, 'stored_path': None})
        except Exception as e:
            st.error(f"身份识别失败: {e}")
            identity_results = [{'is_familiar': False, 'name': None, 'distance': None, 'stored_path': None} for _ in face_objs]

        if auto_store:
            for i, id_res in enumerate(identity_results):
                if not id_res['is_familiar']:
                    try:
                        face_img = face_images[i]
                        face_img_bgr = (face_img * 255).astype("uint8")
                        face_img_bgr = cv2.cvtColor(face_img_bgr, cv2.COLOR_RGB2BGR)
                        filename = f"unknown_{uuid.uuid4().hex}_{i}.jpg"
                        save_path = os.path.join(KNOWN_FACES_DIR, filename)
                        cv2.imwrite(save_path, face_img_bgr)
                        identity_results[i]['stored_path'] = save_path
                    except Exception as e:
                        st.warning(f"存储第{i}张人脸失败: {e}")

    return display_faces, identity_results, attribute_results

# ---------- 摄像头实时视频流（仅面部表情） ----------
def process_camera_stream(stream_source=0, frame_interval=5):
    cap = cv2.VideoCapture(stream_source)
    if not cap.isOpened():
        st.error("无法打开摄像头")
        return
    video_placeholder = st.empty()
    stop_button = st.button("⏹️ 停止摄像头")
    frame_count = 0
    last_results = {}
    while cap.isOpened() and not st.session_state.stop_stream:
        ret, frame = cap.read()
        if not ret:
            st.warning("摄像头断开")
            break
        frame_count += 1
        do_analyze = (frame_count % frame_interval == 0)
        display_frame = frame.copy()
        if do_analyze:
            small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
            temp_path = "temp_cam_frame.jpg"
            cv2.imwrite(temp_path, small_frame)
            try:
                analysis_list = DeepFace.analyze(
                    img_path=temp_path,
                    actions=['emotion'],
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=False,
                    silent=True
                )
                last_results.clear()
                for idx, face_info in enumerate(analysis_list):
                    region = face_info.get('region', {})
                    x = region.get('x', 0) * 2
                    y = region.get('y', 0) * 2
                    w = region.get('w', 0) * 2
                    h = region.get('h', 0) * 2
                    emotion = face_info.get('dominant_emotion', 'unknown')
                    last_results[idx] = {'box': (x, y, w, h), 'emotion': emotion}
            except Exception:
                pass
            if os.path.exists(temp_path):
                os.remove(temp_path)
        for res in last_results.values():
            x, y, w, h = res['box']
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(display_frame, res['emotion'], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(display_frame_rgb, channels="RGB", use_container_width=True)
        if stop_button or st.session_state.stop_stream:
            st.session_state.stop_stream = True
            break
        time.sleep(0.03)
    cap.release()
    st.session_state.streaming = False
    st.session_state.stop_stream = False
    st.success("摄像头已停止")

# ===== 视频文件处理（语音+面部融合） =====
def process_uploaded_video(video_path, frame_interval=5):
    sensevoice_model = load_sensevoice_model()
    speech_result = {"emotion": "unknown", "confidence": 0.0}
    if sensevoice_model:
        try:
            res = sensevoice_model.generate(
                input=video_path,
                language="auto",
                use_itn=True,
                batch_size_s=60,
                merge_vad=True
            )
            if res and len(res) > 0:
                text = rich_transcription_postprocess(res[0]["text"])
                if "<|HAPPY|>" in text:
                    speech_result = {"emotion": "happy", "confidence": 0.8}
                elif "<|SAD|>" in text:
                    speech_result = {"emotion": "sad", "confidence": 0.8}
                elif "<|ANGRY|>" in text:
                    speech_result = {"emotion": "angry", "confidence": 0.8}
                elif "<|NEUTRAL|>" in text:
                    speech_result = {"emotion": "neutral", "confidence": 0.6}
                else:
                    speech_result = {"emotion": "neutral", "confidence": 0.5}
                st.info(f"🎤 语音情绪识别结果：{speech_result['emotion']}")
        except Exception as e:
            st.error(f"语音情感分析失败: {e}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("无法打开视频文件")
        return
    video_placeholder = st.empty()
    stop_button = st.button("⏹️ 停止视频分析")
    frame_count = 0
    last_results = {}
    face_emotions_list = []
    while cap.isOpened() and not st.session_state.stop_stream:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        do_analyze = (frame_count % frame_interval == 0)
        display_frame = frame.copy()
        if do_analyze:
            small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
            temp_path = "temp_vid_frame.jpg"
            cv2.imwrite(temp_path, small_frame)
            try:
                analysis_list = DeepFace.analyze(
                    img_path=temp_path,
                    actions=['emotion'],
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=False,
                    silent=True
                )
                last_results.clear()
                for idx, face_info in enumerate(analysis_list):
                    region = face_info.get('region', {})
                    x = region.get('x', 0) * 2
                    y = region.get('y', 0) * 2
                    w = region.get('w', 0) * 2
                    h = region.get('h', 0) * 2
                    emotion = face_info.get('dominant_emotion', 'unknown')
                    face_emotions_list.append(emotion)
                    last_results[idx] = {'box': (x, y, w, h), 'emotion': emotion}
            except Exception:
                pass
            if os.path.exists(temp_path):
                os.remove(temp_path)
        for res in last_results.values():
            x, y, w, h = res['box']
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(display_frame, res['emotion'], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        if speech_result and speech_result['emotion'] != "unknown":
            text = f"Voice: {speech_result['emotion']}"
            cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(display_frame_rgb, channels="RGB", use_container_width=True)
        if stop_button or st.session_state.stop_stream:
            st.session_state.stop_stream = True
            break
        time.sleep(0.03)
    cap.release()
    st.session_state.stop_stream = False
    if face_emotions_list:
        most_common_face = Counter(face_emotions_list).most_common(1)[0][0]
    else:
        most_common_face = "unknown"
    final_emotion = most_common_face
    if speech_result['confidence'] > 0.7:
        final_emotion = speech_result['emotion']
    st.session_state.fusion_result = {
        "final_emotion": final_emotion,
        "face_emotion": most_common_face,
        "speech_emotion": speech_result['emotion'],
        "speech_confidence": speech_result['confidence']
    }
    st.success(f"🧠 最终情绪判断：{final_emotion}")

# ---------- Streamlit UI ----------
st.set_page_config(page_title="人脸识别与情绪分析系统", layout="wide")
st.title("📸 人脸识别与多模态情绪分析")
st.markdown("基于 YOLO + DeepFace (面部) + SenseVoiceSmall (语音) 实现")

with st.sidebar:
    st.header("📁 已知人脸库")
    if st.button("🖼️ 展示已存储的人脸照片"):
        if os.path.exists(KNOWN_FACES_DIR) and os.listdir(KNOWN_FACES_DIR):
            images = [f for f in os.listdir(KNOWN_FACES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                cols = st.columns(3)
                for idx, img_file in enumerate(images[:9]):
                    with cols[idx % 3]:
                        img_path = os.path.join(KNOWN_FACES_DIR, img_file)
                        st.image(img_path, caption=img_file, use_container_width=True)
            else:
                st.info("暂无照片")
        else:
            st.info("数据库文件夹为空")

# -------------------- 功能1：上传图片识别 --------------------
st.subheader("1️⃣ 选择要识别的图片")
uploaded_file = st.file_uploader("上传一张照片", type=['jpg', 'jpeg', 'png'])
temp_file_path = None
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        tmp.write(uploaded_file.getvalue())
        temp_file_path = tmp.name
    st.image(uploaded_file, caption="已选择的图片", use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    recognize_identity = st.button("🔍 开始识别是否是熟人", disabled=temp_file_path is None, use_container_width=True)
with col2:
    recognize_attributes = st.button("🧬 开始识别表情", disabled=temp_file_path is None, use_container_width=True)

if temp_file_path and (recognize_identity or recognize_attributes):
    with st.spinner("正在分析，请稍候..."):
        display_faces, identity_res, attribute_res = process_multi_faces(temp_file_path, auto_store=True)
    if not display_faces:
        st.warning("未检测到任何人脸")
    else:
        st.success(f"✅ 共检测到 {len(display_faces)} 张人脸")
        for i, face_img in enumerate(display_faces):
            with st.expander(f"👤 人脸 #{i+1}", expanded=True):
                col_img, col_info = st.columns([1, 2])
                with col_img:
                    st.image(face_img, caption=f"人脸 {i+1}", width=150)
                with col_info:
                    if recognize_identity:
                        id_res = identity_res[i] if i < len(identity_res) else None
                        if id_res:
                            if id_res['is_familiar']:
                                st.markdown(f"✅ **身份**: 熟悉的人 - **{id_res['name']}**")
                                st.markdown(f"📏 相似度距离: {id_res['distance']:.4f}")
                            else:
                                st.markdown("❌ **身份**: 陌生人")
                                if id_res.get('stored_path'):
                                    st.info(f"已自动存入数据库: {os.path.basename(id_res['stored_path'])}")
                    if recognize_attributes:
                        attr_res = attribute_res[i] if i < len(attribute_res) else None
                        if attr_res:
                            st.markdown(f"😃 **表情**: {attr_res['emotion']}")
    if temp_file_path and os.path.exists(temp_file_path):
        os.unlink(temp_file_path)

# -------------------- 新增：通过摄像头拍照识别 --------------------
st.markdown("---")
st.subheader("📸 或者使用摄像头拍照识别")
camera_photo = st.camera_input("点击打开摄像头并拍照", key="camera_photo")
if camera_photo is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        tmp.write(camera_photo.getvalue())
        temp_cam_path = tmp.name
    st.image(camera_photo, caption="拍照结果", use_container_width=True)
    col_cam1, col_cam2 = st.columns(2)
    with col_cam1:
        recognize_cam_identity = st.button("🔍 识别是否是熟人（拍照）", key="cam_identity", use_container_width=True)
    with col_cam2:
        recognize_cam_attributes = st.button("🧬 识别表情（拍照）", key="cam_attrs", use_container_width=True)
    if recognize_cam_identity or recognize_cam_attributes:
        with st.spinner("正在分析拍照图片..."):
            display_faces, identity_res, attribute_res = process_multi_faces(temp_cam_path, auto_store=True)
        if not display_faces:
            st.warning("未检测到人脸，请重新拍照")
        else:
            st.success(f"✅ 共检测到 {len(display_faces)} 张人脸")
            for i, face_img in enumerate(display_faces):
                with st.expander(f"👤 人脸 #{i+1}", expanded=True):
                    col_img, col_info = st.columns([1, 2])
                    with col_img:
                        st.image(face_img, caption=f"人脸 {i+1}", width=150)
                    with col_info:
                        if recognize_cam_identity:
                            id_res = identity_res[i] if i < len(identity_res) else None
                            if id_res:
                                if id_res['is_familiar']:
                                    st.markdown(f"✅ **身份**: 熟悉的人 - **{id_res['name']}**")
                                    st.markdown(f"📏 相似度距离: {id_res['distance']:.4f}")
                                else:
                                    st.markdown("❌ **身份**: 陌生人")
                                    if id_res.get('stored_path'):
                                        st.info(f"已自动存入数据库: {os.path.basename(id_res['stored_path'])}")
                        if recognize_cam_attributes:
                            attr_res = attribute_res[i] if i < len(attribute_res) else None
                            if attr_res:
                                st.markdown(f"😃 **表情**: {attr_res['emotion']}")
    if os.path.exists(temp_cam_path):
        os.unlink(temp_cam_path)

# -------------------- 功能2：摄像头实时视频流（仅面部表情） --------------------
st.markdown("---")
st.subheader("2️⃣ 摄像头实时视频流（仅面部表情）")
col_cam1, _ = st.columns([1, 3])
with col_cam1:
    cam_frame_interval = st.slider("分析频率（每N帧分析一次）", min_value=1, max_value=20, value=5, key="cam_interval", help="数值越大，分析越不频繁，但速度越快")
    start_cam = st.button("🎥 启动摄像头", use_container_width=True)
if start_cam:
    st.session_state.stop_stream = False
    st.session_state.streaming = True
    with st.spinner("正在启动摄像头..."):
        process_camera_stream(stream_source=0, frame_interval=cam_frame_interval)

# -------------------- 功能3：上传视频文件（语音+面部融合） --------------------
st.markdown("---")
st.subheader("3️⃣ 上传视频文件分析（语音语调 + 面部表情融合）")
uploaded_video = st.file_uploader("上传视频文件", type=['mp4', 'avi', 'mov'])
if uploaded_video is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_vid:
        tmp_vid.write(uploaded_video.getvalue())
        video_path = tmp_vid.name
    st.video(uploaded_video)
    vid_frame_interval = st.slider("分析频率（每N帧分析一次）", min_value=1, max_value=20, value=5, key="vid_interval")
    start_vid_analysis = st.button("🎬 开始分析视频（语音+面部）", use_container_width=True)
    if start_vid_analysis:
        st.session_state.stop_stream = False
        with st.spinner("正在分析视频，请稍候..."):
            process_uploaded_video(video_path, frame_interval=vid_frame_interval)
        if st.session_state.fusion_result:
            res = st.session_state.fusion_result
            st.markdown("### 📊 综合情绪分析结果")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("🎭 最终情绪", res['final_emotion'])
            with col_b:
                st.metric("😃 面部表情", res['face_emotion'])
            with col_c:
                st.metric("🎤 语音情绪", f"{res['speech_emotion']} (置信度: {res['speech_confidence']:.2f})")
        if os.path.exists(video_path):
            os.unlink(video_path)

# 额外信息
st.sidebar.markdown("---")
if os.path.exists(KNOWN_FACES_DIR):
    num_known = len([f for f in os.listdir(KNOWN_FACES_DIR) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    st.sidebar.info(f"📚 当前已知人脸库中共有 {num_known} 张照片")