import streamlit as st
import cv2
import tempfile
import pandas as pd
import os
from ultralytics import YOLO
import sys

st.write("Python version:", sys.version)

try:
    import cv2
    st.write("OpenCV version:", cv2.__version__)
except Exception as e:
    st.error(f"OpenCV error: {e}")
    st.stop()
# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Video Object Tracking",
    page_icon="🎯",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown(""" 
<style> 
 
/* Main background */ 
.stApp { 
    background: linear-gradient(135deg, #0f172a, #111827, #172554); 
} 
 
/* Main content */ 
.main { 
    padding-top: 1rem; 
} 
 
/* Header */ 
.hero { 
    padding: 25px 30px; 
    border-radius: 18px; 
    margin-bottom: 25px; 
    background: linear-gradient(135deg, #4f46e5, #7c3aed); 
    box-shadow: 0 8px 25px rgba(0,0,0,0.25); 
} 
 
.hero-title { 
    font-size: 34px; 
    font-weight: 800; 
    color: white !important; 
    margin-bottom: 5px; 
} 
 
.hero-subtitle { 
    font-size: 15px; 
    color: white !important; 
} 
 
/* Section titles */ 
.section-title { 
    font-size: 22px; 
    font-weight: 700; 
    color: white !important; 
    margin-top: 20px; 
    margin-bottom: 14px; 
} 
 
/* Metric cards */ 
div[data-testid="stMetric"] { 
    background: linear-gradient(135deg, #1e293b, #312e81); 
    border: 1px solid #4f46e5; 
    padding: 15px; 
    border-radius: 14px; 
    box-shadow: 0 5px 15px rgba(0,0,0,0.2); 
} 
 
/* Metric text */ 
div[data-testid="stMetricLabel"], 
div[data-testid="stMetricValue"], 
div[data-testid="stMetricDelta"] { 
    color: #ffffff !important; 
} 
 
/* Buttons */ 
.stButton > button { 
    border-radius: 12px; 
    font-weight: 700; 
    border: none; 
    background: linear-gradient(135deg, #22c55e, #16a34a); 
    color: white !important; 
    box-shadow: 0 5px 15px rgba(34,197,94,0.3); 
} 
 
.stButton > button:hover { 
    background: linear-gradient(135deg, #16a34a, #15803d); 
    color: white !important; 
} 
 
/* Download button */ 
.stDownloadButton > button { 
    border-radius: 12px; 
    font-weight: 700; 
    background: linear-gradient(135deg, #2563eb, #4f46e5); 
    color: white !important; 
    border: none; 
} 
 
/* File uploader */ 


[data-testid="stFileUploader"] section {
    background: #1e293b !important;
    border: 2px dashed #6366f1 !important;
    border-radius: 15px !important;
    padding: 20px !important;
    transition: all 0.3s ease;
}

/* Hover effect */
[data-testid="stFileUploader"] section:hover {
    background: #273449 !important;
    border-color: #818cf8 !important;
    box-shadow: 0 0 15px rgba(99, 102, 241, 0.25);
}

/* Upload drop-zone text only */
[data-testid="stFileUploader"] section p,
[data-testid="stFileUploader"] section span,
[data-testid="stFileUploader"] section small {
    color: #ffffff !important;
}

/* Browse files button */
[data-testid="stFileUploader"] section button {
    background: #4f46e5 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* Browse button hover */
[data-testid="stFileUploader"] section button:hover {
    background: #6366f1 !important;
}
 
/* Slider */ 
.stSlider > div > div > div > div { 
    background: #6366f1; 
} 
 
/* Radio buttons */ 
.stRadio > div { 
    background: #1e293b; 
    padding: 10px; 
    border-radius: 12px; 
} 
 
/* Select box */ 
.stSelectbox > div > div { 
    border-radius: 10px; 
} 
 
/* Number input */ 
.stNumberInput > div > div { 
    border-radius: 10px; 
} 
 
/* Dataframes */ 
[data-testid="stDataFrame"] { 
    border: 1px solid #3730a3; 
    border-radius: 12px; 
    overflow: hidden; 
} 
 
/* Info messages */ 
.stAlert { 
    border-radius: 12px; 
} 
 
/* Divider */ 
hr { 
    border-color: #3730a3; 
} 


/* =========================================================
   ALL TEXT WHITE
   ========================================================= */

.stApp p,
.stApp span,
.stApp label,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6,
.stApp div,
.stApp small {
    color: #ffffff !important;
}





/* Radio button text */

.stRadio label,
.stRadio label p,
.stRadio label span {
    color: #ffffff !important;
}




/* Number input text */

.stNumberInput label,
.stNumberInput input {
    color: #ffffff !important;
}


/* Slider text */

.stSlider label,
.stSlider span {
    color: #ffffff !important;
}


/* Alert / info / warning / success text */

.stAlert,
.stAlert p,
.stAlert span,
.stAlert div {
    color: #ffffff !important;
}


/* Button text */

.stButton button,
.stButton button p,
.stButton button span {
    color: #ffffff !important;
}


/* Download button text */

.stDownloadButton button,
.stDownloadButton button p,
.stDownloadButton button span {
    color: #ffffff !important;
}


/* Dataframe text */

[data-testid="stDataFrame"] {
    color: #ffffff !important;
}


/* Caption text */

.stCaption,
[data-testid="stCaptionContainer"] {
    color: #ffffff !important;
}


/* Markdown text */

[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span {
    color: #ffffff !important;
}

 
</style> 
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">🎯 AI Video Object Tracking</div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")


model = load_model()


# =========================================================
# SESSION STATE
# =========================================================

if "tracking_output" not in st.session_state:
    st.session_state.tracking_output = None

if "tracking_objects" not in st.session_state:
    st.session_state.tracking_objects = {}

if "processed_frames" not in st.session_state:
    st.session_state.processed_frames = 0

if "tracking_total_frames" not in st.session_state:
    st.session_state.tracking_total_frames = 0

if "tracking_frame_skip" not in st.session_state:
    st.session_state.tracking_frame_skip = 1


# =========================================================
# UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📤 Upload Video",
    type=["mp4", "avi", "mov", "mkv"]
)


if uploaded_file is None:

    st.info("Upload a video to begin.")

    st.stop()


# =========================================================
# SAVE VIDEO
# =========================================================

with tempfile.NamedTemporaryFile(
    delete=False,
    suffix=".mp4"
) as temp:

    temp.write(uploaded_file.getbuffer())
    video_path = temp.name


# =========================================================
# ORIGINAL VIDEO
# =========================================================

st.markdown(
    '<div class="section-title">🎥 Original Video</div>',
    unsafe_allow_html=True
)

st.video(uploaded_file)


# =========================================================
# VIDEO INFORMATION
# =========================================================

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():

    st.error("Could not open video.")
    st.stop()


total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

fps = cap.get(
    cv2.CAP_PROP_FPS
)

if fps <= 0:
    fps = 30

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

duration = (
    total_frames / fps
    if fps > 0
    else 0
)

cap.release()


# =========================================================
# VIDEO INFO
# =========================================================

st.markdown(
    '<div class="section-title">📊 Video Information</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Frames",
    f"{total_frames:,}"
)

c2.metric(
    "FPS",
    f"{fps:.2f}"
)

c3.metric(
    "Duration",
    f"{duration:.2f}s"
)

c4.metric(
    "Resolution",
    f"{width} × {height}"
)


# =========================================================
# DETECTION
# =========================================================

st.markdown(
    '<div class="section-title">🔍 Object Detection</div>',
    unsafe_allow_html=True
)

frame_number_detection = st.slider(
    "Frame",
    0,
    max(total_frames - 1, 0),
    0
)


cap = cv2.VideoCapture(video_path)

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    frame_number_detection
)

success, frame = cap.read()

cap.release()


if success:

    results = model(
        frame,
        conf=0.25,
        imgsz=640,
        verbose=False
    )

    result = results[0]

    annotated_frame = frame.copy()

    try:

        plotted = result.plot()

        if plotted is not None:
            annotated_frame = plotted

    except Exception:
        pass

    try:

        annotated_frame = cv2.cvtColor(
            annotated_frame,
            cv2.COLOR_BGR2RGB
        )

    except Exception:
        pass

    st.image(
        annotated_frame,
        use_container_width=True
    )


    detected_objects = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            detected_objects.append({
                "Object": model.names[class_id],
                "Confidence": f"{confidence * 100:.2f}%"
            })


    if detected_objects:

        detection_df = pd.DataFrame(
            detected_objects
        )

        st.dataframe(
            detection_df,
            use_container_width=True,
            hide_index=True
        )

        st.metric(
            "Objects Detected",
            len(detected_objects)
        )

else:

    st.error("Could not read selected frame.")


# =========================================================
# TRACKING
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">🎯 Object Tracking</div>',
    unsafe_allow_html=True
)


# =========================================================
# SETTINGS
# =========================================================

c1, c2, c3 = st.columns(3)

with c1:

    confidence_threshold = st.slider(
        "Confidence",
        0.10,
        0.90,
        0.25,
        0.05
    )


with c2:

    frame_skip = st.slider(
        "Frame Skip",
        1,
        5,
        1,
        1
    )


with c3:

    imgsz = st.selectbox(
        "Image Size",
        [416, 640, 832],
        index=1
    )


# =========================================================
# TRACKING MODE
# =========================================================

tracking_mode = st.radio(
    "Mode",
    [
        "Show All Objects",
        "Highlight One Object"
    ],
    horizontal=True
)


selected_object_id = None

if tracking_mode == "Highlight One Object":

    selected_object_id = st.number_input(
        "Object ID",
        min_value=1,
        value=1,
        step=1
    )


# =========================================================
# START TRACKING
# =========================================================

start_tracking = st.button(
    "🚀 Start Tracking",
    type="primary",
    use_container_width=True
)


if start_tracking:

    # -----------------------------------------------------
    # RESET PREVIOUS RESULT
    # -----------------------------------------------------

    st.session_state.tracking_output = None
    st.session_state.tracking_objects = {}

    # -----------------------------------------------------
    # OPEN VIDEO
    # -----------------------------------------------------

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        st.error("Could not open video.")
        st.stop()


    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30


    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )


    # -----------------------------------------------------
    # OUTPUT FILE
    # -----------------------------------------------------

    output_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    output_path = output_file.name

    output_file.close()


    # -----------------------------------------------------
    # VIDEO WRITER
    # -----------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps / frame_skip,
        (width, height)
    )


    if not writer.isOpened():

        cap.release()

        st.error(
            "Could not create output video."
        )

        st.stop()


    # -----------------------------------------------------
    # UI
    # -----------------------------------------------------

    progress = st.progress(0)

    status = st.empty()

    video_placeholder = st.empty()


    # -----------------------------------------------------
    # TRACKING DATA
    # -----------------------------------------------------

    custom_ids = {}

    next_custom_id = 1

    all_objects = {}

    frame_number = 0

    processed_frames = 0


    # =====================================================
    # TRACKING LOOP
    # =====================================================

    while True:

        success, frame = cap.read()

        if not success:
            break


        frame_number += 1


        if frame_number % frame_skip != 0:
            continue


        processed_frames += 1


        status.write(
            f"Processing {frame_number:,} / "
            f"{total_frames:,}"
        )


        # -------------------------------------------------
        # YOLO TRACK
        # -------------------------------------------------

        results = model.track(
            frame,
            persist=True,
            conf=confidence_threshold,
            imgsz=imgsz,
            tracker="bytetrack.yaml",
            verbose=False
        )


        result = results[0]

        annotated_frame = frame.copy()


        # -------------------------------------------------
        # TRACKING BOXES
        # -------------------------------------------------

        if (
            result.boxes is not None
            and result.boxes.id is not None
        ):

            boxes = (
                result.boxes.xyxy
                .cpu()
                .numpy()
            )

            track_ids = (
                result.boxes.id
                .cpu()
                .numpy()
                .astype(int)
            )

            classes = (
                result.boxes.cls
                .cpu()
                .numpy()
                .astype(int)
            )

            confidences = (
                result.boxes.conf
                .cpu()
                .numpy()
            )


            # -------------------------------------------------
            # EVERY OBJECT
            # -------------------------------------------------

            for (
                box,
                track_id,
                class_id,
                confidence
            ) in zip(
                boxes,
                track_ids,
                classes,
                confidences
            ):

                # ---------------------------------------------
                # CUSTOM ID
                # ---------------------------------------------

                if track_id not in custom_ids:

                    custom_ids[
                        track_id
                    ] = next_custom_id

                    next_custom_id += 1


                custom_id = custom_ids[
                    track_id
                ]


                # ---------------------------------------------
                # CLASS
                # ---------------------------------------------

                object_name = model.names[
                    class_id
                ]


                # ---------------------------------------------
                # SAVE OBJECT
                # ---------------------------------------------

                if custom_id not in all_objects:

                    all_objects[
                        custom_id
                    ] = {

                        "ID":
                            custom_id,

                        "Object":
                            object_name,

                        "Max Confidence":
                            float(
                                confidence
                            ),

                        "Frames Seen":
                            1

                    }

                else:

                    all_objects[
                        custom_id
                    ]["Frames Seen"] += 1


                    if confidence > all_objects[
                        custom_id
                    ]["Max Confidence"]:

                        all_objects[
                            custom_id
                        ]["Max Confidence"] = float(
                            confidence
                        )


                # ---------------------------------------------
                # COORDINATES
                # ---------------------------------------------

                x1, y1, x2, y2 = map(
                    int,
                    box
                )


                # ---------------------------------------------
                # SELECTED
                # ---------------------------------------------

                is_selected = (

                    tracking_mode
                    == "Highlight One Object"

                    and custom_id
                    == int(
                        selected_object_id
                    )

                )


                # ---------------------------------------------
                # STYLE
                # ---------------------------------------------

                if is_selected:

                    box_color = (
                        0,
                        255,
                        0
                    )

                    thickness = 4

                    label = (
                        f"SELECTED | "
                        f"ID {custom_id} | "
                        f"{object_name} | "
                        f"{confidence * 100:.1f}%"
                    )

                elif tracking_mode == "Highlight One Object":

                    box_color = (
                        160,
                        160,
                        160
                    )

                    thickness = 1

                    label = (
                        f"ID {custom_id} | "
                        f"{object_name}"
                    )

                else:

                    box_color = (
                        0,
                        255,
                        0
                    )

                    thickness = 2

                    label = (
                        f"ID {custom_id} | "
                        f"{object_name} | "
                        f"{confidence * 100:.1f}%"
                    )


                # ---------------------------------------------
                # BOX
                # ---------------------------------------------

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    thickness
                )


                # ---------------------------------------------
                # LABEL
                # ---------------------------------------------

                text_size = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    2
                )[0]

                text_width = text_size[0]

                text_height = text_size[1]

                label_y1 = max(
                    0,
                    y1 - text_height - 10
                )

                label_y2 = max(
                    text_height + 10,
                    y1
                )


                cv2.rectangle(
                    annotated_frame,
                    (
                        x1,
                        label_y1
                    ),
                    (
                        x1 + text_width + 10,
                        label_y2
                    ),
                    box_color,
                    -1
                )


                cv2.putText(
                    annotated_frame,
                    label,
                    (
                        x1 + 5,
                        max(
                            15,
                            y1 - 5
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2
                )


        # -------------------------------------------------
        # WRITE VIDEO
        # -------------------------------------------------

        writer.write(
            annotated_frame
        )


        # -------------------------------------------------
        # LIVE PREVIEW
        # -------------------------------------------------

        try:

            preview_frame = cv2.cvtColor(
                annotated_frame,
                cv2.COLOR_BGR2RGB
            )

            video_placeholder.image(
                preview_frame,
                use_container_width=True
            )

        except Exception:
            pass


        # -------------------------------------------------
        # PROGRESS
        # -------------------------------------------------

        if total_frames > 0:

            progress.progress(
                min(
                    frame_number / total_frames,
                    1.0
                )
            )


    # =====================================================
    # RELEASE
    # =====================================================

    cap.release()

    writer.release()

    progress.progress(1.0)

    status.success(
        "Tracking completed successfully."
    )


    # =====================================================
    # SESSION STATE
    # =====================================================

    st.session_state.tracking_output = (
        output_path
    )

    st.session_state.tracking_objects = (
        all_objects
    )

    st.session_state.processed_frames = (
        processed_frames
    )

    st.session_state.tracking_total_frames = (
        total_frames
    )

    st.session_state.tracking_frame_skip = (
        frame_skip
    )


# =========================================================
# TRACKING OUTPUT
# =========================================================

output_path = st.session_state.tracking_output


if (
    output_path
    and os.path.exists(output_path)
):

    st.divider()

    st.markdown(
        '<div class="section-title">🎥 Tracking Result</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # VIDEO
    # -----------------------------------------------------

    with open(
        output_path,
        "rb"
    ) as video_file:

        video_bytes = video_file.read()


    st.video(
        video_bytes
    )


    # -----------------------------------------------------
    # DOWNLOAD
    # -----------------------------------------------------

    st.download_button(
        "⬇️ Download Tracking Video",
        data=video_bytes,
        file_name="tracked_video.mp4",
        mime="video/mp4",
        use_container_width=True
    )


    # =====================================================
    # OBJECTS
    # =====================================================

    all_objects = st.session_state.tracking_objects


    if all_objects:

        st.markdown(
            '<div class="section-title">📊 Tracked Objects</div>',
            unsafe_allow_html=True
        )


        object_list = []


        for obj in all_objects.values():

            object_list.append({

                "ID":
                    obj["ID"],

                "Object":
                    obj["Object"],

                "Confidence":
                    f"{obj['Max Confidence'] * 100:.2f}%",

                "Frames Seen":
                    obj["Frames Seen"]

            })


        object_df = pd.DataFrame(
            object_list
        )


        object_df = object_df.sort_values(
            by="ID"
        )


        st.dataframe(
            object_df,
            use_container_width=True,
            hide_index=True
        )


        # =================================================
        # STATISTICS
        # =================================================

        st.markdown(
            '<div class="section-title">📈 Statistics</div>',
            unsafe_allow_html=True
        )


        c1, c2, c3, c4 = st.columns(4)


        c1.metric(
            "Unique Objects",
            len(all_objects)
        )

        c2.metric(
            "Total Frames",
            st.session_state.tracking_total_frames
        )

        c3.metric(
            "Processed",
            st.session_state.processed_frames
        )

        c4.metric(
            "Frame Skip",
            st.session_state.tracking_frame_skip
        )


        # =================================================
        # OBJECT TYPE SUMMARY
        # =================================================

        object_counts = {}


        for obj in all_objects.values():

            name = obj["Object"]

            object_counts[name] = (
                object_counts.get(name, 0) + 1
            )


        type_data = [

            {
                "Object": name,
                "Unique Count": count
            }

            for name, count
            in object_counts.items()

        ]


        type_df = pd.DataFrame(
            type_data
        )


        st.dataframe(
            type_df,
            use_container_width=True,
            hide_index=True
        )

else:

    if not start_tracking:
        pass