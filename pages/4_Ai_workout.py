# ============================================================
# FITQUEST AI - AI WORKOUT
# Render-friendly version
# ============================================================

import math
import threading
import time

import av
import cv2
import mediapipe as mp
import streamlit as st

from streamlit_webrtc import (
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

from utils.auth import (
    initialize_auth,
    get_current_user_id,
    is_logged_in,
)

from utils.database import get_user
from utils.gamification import complete_workout
from utils.ui import apply_fitquest_theme


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Workout | FitQuest AI",
    page_icon="🏋️",
    layout="wide",
)

apply_fitquest_theme()
initialize_auth()


# ============================================================
# LOGIN CHECK
# ============================================================

if not is_logged_in():

    st.title("🏋️ AI Workout Arena")

    st.warning(
        "🔐 Please log in before starting an AI workout."
    )

    st.info(
        """
        Your workout progress is connected to your FitQuest account.

        Log in to:

        • Save workout history
        • Earn XP
        • Increase your level
        • Build your streak
        • Update your dashboard
        • Climb the leaderboard
        """
    )

    st.stop()


# ============================================================
# CURRENT USER
# ============================================================

current_user_id = get_current_user_id()

user = get_user(current_user_id)

if user is None:

    st.error(
        "Your FitQuest account could not be loaded."
    )

    st.warning(
        "Please log out and log in again."
    )

    st.stop()


# ============================================================
# MEDIAPIPE
# ============================================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ============================================================
# SESSION STATE
# ============================================================

if "selected_exercise" not in st.session_state:
    st.session_state.selected_exercise = "Bicep Curl"

if "last_workout_result" not in st.session_state:
    st.session_state.last_workout_result = None

if "saved_workout_key" not in st.session_state:
    st.session_state.saved_workout_key = None


# ============================================================
# ANGLE CALCULATION
# ============================================================

def calculate_angle(a, b, c):

    ab = (
        a[0] - b[0],
        a[1] - b[1],
    )

    cb = (
        c[0] - b[0],
        c[1] - b[1],
    )

    denominator = (
        math.hypot(*ab)
        *
        math.hypot(*cb)
    )

    if denominator == 0:
        return 0.0

    cosine = (
        ab[0] * cb[0]
        +
        ab[1] * cb[1]
    ) / denominator

    cosine = max(
        -1.0,
        min(1.0, cosine)
    )

    return math.degrees(
        math.acos(cosine)
    )


# ============================================================
# BICEP CURL ANALYSIS
# ============================================================

def analyze_bicep_curl(angle):

    if angle <= 60:

        return (
            100,
            "Strong Curl",
            "Excellent contraction. Now extend your arm."
        )

    elif angle <= 90:

        return (
            95,
            "Curling",
            "Good movement. Continue upward."
        )

    elif angle <= 120:

        return (
            90,
            "Mid Movement",
            "Keep curling smoothly."
        )

    elif angle <= 150:

        return (
            88,
            "Returning",
            "Extend your arm with control."
        )

    else:

        return (
            90,
            "Ready",
            "Arm is extended. Start your curl."
        )


# ============================================================
# PUSH-UP ANALYSIS
# ============================================================

def analyze_push_up(angle):

    if angle <= 90:

        return (
            100,
            "Bottom Position",
            "Good depth. Push upward."
        )

    elif angle <= 110:

        return (
            95,
            "Good Depth",
            "Good depth. Start pushing upward."
        )

    elif angle <= 135:

        return (
            90,
            "Mid Movement",
            "Continue smoothly."
        )

    elif angle <= 160:

        return (
            88,
            "Pushing Up",
            "Continue pushing upward."
        )

    else:

        return (
            90,
            "Up Position",
            "Arms extended. Lower yourself."
        )


# ============================================================
# POSE PROCESSOR
# ============================================================

class PoseVideoProcessor(VideoProcessorBase):

    def __init__(self, exercise_name):

        self.exercise_name = exercise_name

        self.pose = mp_pose.Pose(

            static_image_mode=False,

            model_complexity=1,

            smooth_landmarks=True,

            enable_segmentation=False,

            min_detection_confidence=0.35,

            min_tracking_confidence=0.35,
        )

        self.reps = 0

        self.current_angle = 0.0

        self.current_score = 0.0

        self.current_status = (
            "Starting camera..."
        )

        self.current_feedback = (
            "Allow camera access and move into view."
        )

        self.landmarks_detected = False

        self.stage = "waiting"

        self.last_rep_time = 0.0

        self.minimum_rep_interval = 0.60

        self.angle_history = []

        self.total_score = 0.0

        self.score_samples = 0

        self.start_time = time.time()

        self.state_lock = threading.Lock()


    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        try:

            self.pose.close()

        except Exception:

            pass


    # ========================================================
    # SMOOTH ANGLE
    # ========================================================

    def smooth_angle(self, angle):

        self.angle_history.append(
            float(angle)
        )

        if len(self.angle_history) > 5:

            self.angle_history.pop(0)

        return (
            sum(self.angle_history)
            /
            len(self.angle_history)
        )


    # ========================================================
    # REGISTER REP
    # ========================================================

    def register_rep(self):

        now = time.time()

        if (
            now - self.last_rep_time
            <
            self.minimum_rep_interval
        ):

            return False

        self.reps += 1

        self.last_rep_time = now

        self.total_score += (
            self.current_score
        )

        self.score_samples += 1

        return True


    # ========================================================
    # SNAPSHOT
    # ========================================================

    def snapshot(self):

        with self.state_lock:

            return {

                "reps": int(
                    self.reps
                ),

                "current_angle": round(
                    float(
                        self.current_angle
                    ),
                    1,
                ),

                "current_score": round(
                    float(
                        self.current_score
                    ),
                    1,
                ),

                "stage": self.stage,

                "current_status":
                    self.current_status,

                "current_feedback":
                    self.current_feedback,

                "landmarks_detected":
                    bool(
                        self.landmarks_detected
                    ),

                "total_score":
                    float(
                        self.total_score
                    ),

                "score_samples":
                    int(
                        self.score_samples
                    ),

                "start_time":
                    float(
                        self.start_time
                    ),
            }


    # ========================================================
    # BICEP REP COUNTING
    # ========================================================

    def count_bicep_curl(self, angle):

        # Waiting for initial straight arm
        if self.stage == "waiting":

            if angle >= 125:

                self.stage = "ready"

                self.current_status = (
                    "Ready for Curl"
                )

                self.current_feedback = (
                    "Good starting position. Curl upward."
                )

            else:

                self.current_status = (
                    "Get Ready"
                )

                self.current_feedback = (
                    "Keep your arm more extended."
                )

            return


        # Starting curl
        if self.stage == "ready":

            if angle <= 120:

                self.stage = "curling"

                self.current_status = (
                    "Curling"
                )

                self.current_feedback = (
                    "Good. Keep curling upward."
                )

            return


        # Curl reaches top
        if self.stage == "curling":

            if angle <= 100:

                self.stage = "top"

                self.current_status = (
                    "Top Position"
                )

                self.current_feedback = (
                    "Great contraction. Lower your arm."
                )

            return


        # Returning arm
        if self.stage == "top":

            if angle >= 115:

                self.stage = "returning"

                self.current_status = (
                    "Returning"
                )

                self.current_feedback = (
                    "Keep extending your arm."
                )

            return


        # Complete rep
        if self.stage == "returning":

            if angle >= 125:

                if self.register_rep():

                    self.stage = "ready"

                    self.current_status = (
                        "Rep Counted"
                    )

                    self.current_feedback = (
                        "Excellent! Full repetition counted."
                    )


    # ========================================================
    # PUSH-UP REP COUNTING
    # ========================================================

    def count_push_up(self, angle):

        if self.stage == "waiting":

            if angle >= 145:

                self.stage = "ready"

                self.current_status = (
                    "Ready for Push Up"
                )

                self.current_feedback = (
                    "Good starting position. Lower yourself."
                )

            else:

                self.current_status = (
                    "Get Ready"
                )

                self.current_feedback = (
                    "Move into a straighter position."
                )

            return


        if self.stage == "ready":

            if angle <= 135:

                self.stage = "lowering"

                self.current_status = (
                    "Lowering"
                )

                self.current_feedback = (
                    "Good. Continue lowering."
                )

            return


        if self.stage == "lowering":

            if angle <= 110:

                self.stage = "bottom"

                self.current_status = (
                    "Bottom Position"
                )

                self.current_feedback = (
                    "Good depth. Push upward."
                )

            return


        if self.stage == "bottom":

            if angle >= 130:

                self.stage = "pushing"

                self.current_status = (
                    "Pushing Up"
                )

                self.current_feedback = (
                    "Push back to the upper position."
                )

            return


        if self.stage == "pushing":

            if angle >= 145:

                if self.register_rep():

                    self.stage = "ready"

                    self.current_status = (
                        "Rep Counted"
                    )

                    self.current_feedback = (
                        "Excellent! Full push-up counted."
                    )


    # ========================================================
    # VIDEO FRAME
    # ========================================================

    def recv(self, frame):

        image = frame.to_ndarray(
            format="bgr24"
        )

        # Mirror camera
        image = cv2.flip(
            image,
            1
        )

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # MEDIAPIPE
        # ----------------------------------------------------

        try:

            results = self.pose.process(
                rgb
            )

        except Exception:

            with self.state_lock:

                self.current_status = (
                    "AI Processing Error"
                )

                self.current_feedback = (
                    "Camera is connected, but AI processing failed."
                )

                self.landmarks_detected = False

            cv2.putText(
                image,
                "AI PROCESSING ERROR",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2,
            )

            return av.VideoFrame.from_ndarray(
                image,
                format="bgr24"
            )


        with self.state_lock:

            self.landmarks_detected = False


        # ----------------------------------------------------
        # POSE FOUND
        # ----------------------------------------------------

        if results.pose_landmarks:

            landmarks = (
                results.pose_landmarks.landmark
            )

            right_ids = [

                mp_pose.PoseLandmark.RIGHT_SHOULDER,

                mp_pose.PoseLandmark.RIGHT_ELBOW,

                mp_pose.PoseLandmark.RIGHT_WRIST,
            ]

            left_ids = [

                mp_pose.PoseLandmark.LEFT_SHOULDER,

                mp_pose.PoseLandmark.LEFT_ELBOW,

                mp_pose.PoseLandmark.LEFT_WRIST,
            ]


            right = [
                landmarks[i.value]
                for i in right_ids
            ]

            left = [
                landmarks[i.value]
                for i in left_ids
            ]


            right_visibility = min(
                float(p.visibility)
                for p in right
            )

            left_visibility = min(
                float(p.visibility)
                for p in left
            )


            if right_visibility >= left_visibility:

                points = right

                best_visibility = (
                    right_visibility
                )

            else:

                points = left

                best_visibility = (
                    left_visibility
                )


            # ------------------------------------------------
            # VISIBILITY CHECK
            # ------------------------------------------------

            if best_visibility >= 0.30:

                height, width = (
                    image.shape[:2]
                )

                shoulder = points[0]

                elbow = points[1]

                wrist = points[2]


                shoulder_pt = (
                    shoulder.x * width,
                    shoulder.y * height,
                )

                elbow_pt = (
                    elbow.x * width,
                    elbow.y * height,
                )

                wrist_pt = (
                    wrist.x * width,
                    wrist.y * height,
                )


                raw_angle = calculate_angle(
                    shoulder_pt,
                    elbow_pt,
                    wrist_pt,
                )


                angle = self.smooth_angle(
                    raw_angle
                )


                # --------------------------------------------
                # EXERCISE ANALYSIS
                # --------------------------------------------

                if (
                    self.exercise_name
                    ==
                    "Bicep Curl"
                ):

                    (
                        score,
                        status,
                        feedback,
                    ) = analyze_bicep_curl(
                        angle
                    )

                else:

                    (
                        score,
                        status,
                        feedback,
                    ) = analyze_push_up(
                        angle
                    )


                with self.state_lock:

                    self.landmarks_detected = True

                    self.current_angle = angle

                    self.current_score = score

                    self.current_status = status

                    self.current_feedback = feedback


                    if (
                        self.exercise_name
                        ==
                        "Bicep Curl"
                    ):

                        self.count_bicep_curl(
                            angle
                        )

                    else:

                        self.count_push_up(
                            angle
                        )


                # --------------------------------------------
                # ANGLE ON SCREEN
                # --------------------------------------------

                cv2.putText(

                    image,

                    f"ANGLE: {int(angle)}",

                    (
                        int(
                            elbow.x * width
                        ) + 10,

                        int(
                            elbow.y * height
                        ) - 10,
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.70,

                    (0, 255, 0),

                    2,
                )


            else:

                with self.state_lock:

                    self.current_status = (
                        "Move Into View"
                    )

                    self.current_feedback = (
                        "Keep shoulder, elbow and wrist visible."
                    )


            # ------------------------------------------------
            # DRAW LANDMARKS
            # ------------------------------------------------

            mp_drawing.draw_landmarks(

                image,

                results.pose_landmarks,

                mp_pose.POSE_CONNECTIONS,

                landmark_drawing_spec=(
                    mp_drawing_styles
                    .get_default_pose_landmarks_style()
                ),
            )


        else:

            with self.state_lock:

                self.current_status = (
                    "No Pose Detected"
                )

                self.current_feedback = (
                    "Move into the camera view."
                )


        # ====================================================
        # OVERLAY
        # ====================================================

        snap = self.snapshot()


        cv2.rectangle(

            image,

            (10, 10),

            (390, 190),

            (0, 0, 0),

            -1,
        )


        cv2.putText(

            image,

            f"REPS: {snap['reps']}",

            (25, 48),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.85,

            (0, 255, 0),

            2,
        )


        cv2.putText(

            image,

            f"ANGLE: {int(snap['current_angle'])}",

            (25, 85),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 255, 255),

            2,
        )


        cv2.putText(

            image,

            f"STAGE: {snap['stage']}",

            (25, 120),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 255, 255),

            2,
        )


        cv2.putText(

            image,

            f"SCORE: {int(snap['current_score'])}",

            (25, 155),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 255, 255),

            2,
        )


        return av.VideoFrame.from_ndarray(

            image,

            format="bgr24"
        )


# ============================================================
# PAGE HEADER
# ============================================================

st.title(
    "🏋️ AI Workout Arena"
)

st.write(
    f"Welcome, **{user.get('username', 'FitQuest Player')}**! "
    "Use real-time AI pose detection to analyze your exercise."
)


# ============================================================
# USER STATS
# ============================================================

columns = st.columns(4)


columns[0].metric(
    "⭐ Current XP",
    user.get("xp", 0)
)

columns[1].metric(
    "🏆 Level",
    user.get("level", 1)
)

columns[2].metric(
    "🔥 Streak",
    f"{user.get('streak', 0)} Days"
)

columns[3].metric(
    "💪 Workouts",
    user.get("total_workouts", 0)
)


st.divider()


# ============================================================
# EXERCISE SELECTION
# ============================================================

selected_exercise = st.selectbox(

    "🏋️ Choose Exercise",

    [
        "Bicep Curl",
        "Push Up",
    ],

    index=(
        0
        if st.session_state.selected_exercise
        ==
        "Bicep Curl"
        else 1
    ),

    key="exercise_selector",
)


st.session_state.selected_exercise = (
    selected_exercise
)


# ============================================================
# INSTRUCTIONS
# ============================================================

if selected_exercise == "Bicep Curl":

    st.info(
        """
        📷 **Bicep Curl Setup**

        • Stand side-on to the camera.
        • Keep your full arm visible.
        • Shoulder, elbow and wrist should be visible.
        • Start with your arm mostly straight.
        • Curl upward.
        • Return to the straight position.
        """
    )

else:

    st.info(
        """
        📷 **Push Up Setup**

        • Place the camera side-on.
        • Keep shoulder, elbow and wrist visible.
        • Start with arms extended.
        • Lower your body.
        • Push back up.
        """
    )


# ============================================================
# CAMERA
# ============================================================

st.subheader(
    "📷 Live AI Camera"
)

st.caption(
    "Allow camera permission when your browser asks."
)

st.caption(
    "Chrome or Edge recommended."
)


# ============================================================
# WEBRTC CONFIGURATION
# ============================================================
#
# IMPORTANT FOR STREAMLIT COMMUNITY CLOUD:
# STUN alone is not reliable for every network.
# We therefore support a TURN server through Streamlit Secrets.
#
# The app still keeps several public STUN servers as a fallback.
# TURN credentials are NEVER hard-coded in this file.
#
# Add these to Streamlit Cloud -> App Settings -> Secrets:
#
# TURN_USERNAME = "your_turn_username"
# TURN_CREDENTIAL = "your_turn_credential"
#
# The default URLs below are for the free Open Relay/Metered
# TURN service. The free tier is limited, so use it for testing
# and normal personal/demo usage.
#
# ============================================================

def get_ice_servers():
    """
    Build the ICE server list used by streamlit-webrtc.

    STUN is always included.
    TURN is added only when TURN_USERNAME and TURN_CREDENTIAL
    are available in Streamlit Secrets.
    """

    ice_servers = [
        {
            "urls": [
                "stun:stun.l.google.com:19302",
            ]
        },
        {
            "urls": [
                "stun:stun1.l.google.com:19302",
            ]
        },
        {
            "urls": [
                "stun:stun2.l.google.com:19302",
            ]
        },
        {
            "urls": [
                "stun:stun.cloudflare.com:3478",
            ]
        },
    ]

    # --------------------------------------------------------
    # Read TURN credentials from Streamlit Secrets
    # --------------------------------------------------------

    try:
        turn_username = str(
            st.secrets.get("TURN_USERNAME", "")
        ).strip()

        turn_credential = str(
            st.secrets.get("TURN_CREDENTIAL", "")
        ).strip()

        # Optional custom TURN server.
        # If not supplied, use the free Metered/OpenRelay
        # endpoints.
        turn_server = str(
            st.secrets.get(
                "TURN_SERVER",
                "standard.relay.metered.ca",
            )
        ).strip()

    except Exception:
        turn_username = ""
        turn_credential = ""
        turn_server = "standard.relay.metered.ca"

    # --------------------------------------------------------
    # Add TURN only when credentials exist
    # --------------------------------------------------------

    if turn_username and turn_credential:

        ice_servers.extend(
            [
                {
                    "urls": [
                        f"turn:{turn_server}:80",
                    ],
                    "username": turn_username,
                    "credential": turn_credential,
                },
                {
                    "urls": [
                        f"turn:{turn_server}:80?transport=tcp",
                    ],
                    "username": turn_username,
                    "credential": turn_credential,
                },
                {
                    "urls": [
                        f"turn:{turn_server}:443",
                    ],
                    "username": turn_username,
                    "credential": turn_credential,
                },
                {
                    "urls": [
                        f"turns:{turn_server}:443?transport=tcp",
                    ],
                    "username": turn_username,
                    "credential": turn_credential,
                },
            ]
        )

    return ice_servers


ICE_SERVERS = get_ice_servers()

TURN_ENABLED = any(
    isinstance(server, dict)
    and server.get("username")
    and server.get("credential")
    for server in ICE_SERVERS
)


# ============================================================
# CAMERA STREAM
# ============================================================

exercise_key = (
    selected_exercise
    .lower()
    .replace(" ", "-")
)

try:

    webrtc_ctx = webrtc_streamer(

        key=(
            f"fitquest-ai-camera-"
            f"{current_user_id}-"
            f"{exercise_key}"
        ),

        mode=WebRtcMode.SENDRECV,

        media_stream_constraints={

            "video": {

                "width": {
                    "ideal": 640,
                    "max": 1280,
                },

                "height": {
                    "ideal": 480,
                    "max": 720,
                },

                "frameRate": {
                    "ideal": 20,
                    "max": 30,
                },

                "facingMode": "user",
            },

            "audio": False,
        },

        video_processor_factory=lambda:
            PoseVideoProcessor(
                selected_exercise
            ),

        async_processing=True,

        # THIS IS THE IMPORTANT PART:
        # Pass STUN + TURN servers to WebRTC.
        rtc_configuration={
            "iceServers": ICE_SERVERS
        },
    )


except Exception as error:

    st.error(
        "❌ Camera component could not start."
    )

    st.code(
        str(error)
    )

    st.info(
        """
        Try these steps:

        1. Use Chrome or Edge.
        2. Allow camera permission.
        3. Refresh the page.
        4. Close other apps using the camera.
        5. Try again.
        """
    )

    st.stop()


# ============================================================
# CONNECTION STATUS
# ============================================================

if webrtc_ctx.state.playing:

    if TURN_ENABLED:

        st.success(
            "🟢 Camera connected successfully using WebRTC + TURN."
        )

    else:

        st.success(
            "🟢 Camera connected successfully!"
        )

    st.info(
        "Move into the camera view and start your exercise."
    )

elif webrtc_ctx.state.signalling:

    st.info(
        "🔄 Connecting to your camera... "
        "Please wait a few seconds."
    )

else:

    st.info(
        "Click START above the camera to begin."
    )


# ============================================================
# LIVE AI ANALYSIS
# ============================================================

if hasattr(st, "fragment"):

    @st.fragment(run_every="500ms")
    def live_analysis():

        processor = (
            webrtc_ctx.video_processor
        )

        if processor:

            live = processor.snapshot()

            st.divider()

            st.subheader(
                "🤖 Live AI Analysis"
            )

            m1, m2, m3, m4 = (
                st.columns(4)
            )

            m1.metric(
                "🔁 Repetitions",
                live["reps"]
            )

            m2.metric(
                "📐 Elbow Angle",
                f"{live['current_angle']}°"
            )

            m3.metric(
                "⭐ Form Score",
                f"{live['current_score']:.0f}/100"
            )

            m4.metric(
                "🔄 Stage",
                live["stage"].title()
            )

            st.info(
                f"**Status:** "
                f"{live['current_status']} — "
                f"{live['current_feedback']}"
            )


            if live["landmarks_detected"]:

                st.success(
                    "✅ Body landmarks detected."
                )

            else:

                st.warning(
                    "👤 Waiting for a clear body pose..."
                )

        else:

            st.caption(
                "AI analysis will appear after the camera starts."
            )


    live_analysis()


else:

    processor = (
        webrtc_ctx.video_processor
    )

    if processor:

        live = processor.snapshot()

        st.divider()

        st.subheader(
            "🤖 Live AI Analysis"
        )

        m1, m2, m3, m4 = (
            st.columns(4)
        )

        m1.metric(
            "🔁 Repetitions",
            live["reps"]
        )

        m2.metric(
            "📐 Elbow Angle",
            f"{live['current_angle']}°"
        )

        m3.metric(
            "⭐ Form Score",
            f"{live['current_score']:.0f}/100"
        )

        m4.metric(
            "🔄 Stage",
            live["stage"].title()
        )

        st.info(
            f"**Status:** "
            f"{live['current_status']} — "
            f"{live['current_feedback']}"
        )


# ============================================================
# COMPLETE WORKOUT
# ============================================================

st.divider()

st.subheader(
    "🏁 Complete Workout"
)

st.write(
    "Complete at least one full repetition before saving."
)


if st.button(
    "🏆 Complete and Save Workout",
    use_container_width=True,
):

    processor = (
        webrtc_ctx.video_processor
    )


    if not processor:

        st.error(
            "Please start the AI camera first."
        )


    else:

        workout_state = (
            processor.snapshot()
        )

        reps = int(
            workout_state["reps"]
        )


        if workout_state[
            "score_samples"
        ]:

            form_score = round(

                workout_state[
                    "total_score"
                ]
                /
                workout_state[
                    "score_samples"
                ],

                1,
            )

        else:

            form_score = round(

                workout_state[
                    "current_score"
                ],

                1,
            )


        # ----------------------------------------------------
        # NO REPS
        # ----------------------------------------------------

        if reps <= 0:

            st.warning(
                "No completed repetition detected yet."
            )

            st.info(
                "Perform one complete repetition and try again."
            )


        else:

            workout_key = (

                f"{current_user_id}-"
                f"{selected_exercise}-"
                f"{reps}-"
                f"{workout_state['start_time']}"
            )


            # ------------------------------------------------
            # PREVENT DUPLICATE SAVE
            # ------------------------------------------------

            if (
                st.session_state.saved_workout_key
                ==
                workout_key
            ):

                st.warning(
                    "This workout has already been saved."
                )


            else:

                try:

                    result = complete_workout(

                        user_id=current_user_id,

                        exercise_name=(
                            selected_exercise
                        ),

                        reps=reps,

                        form_score=form_score,
                    )


                    st.session_state.last_workout_result = (
                        result
                    )

                    st.session_state.saved_workout_key = (
                        workout_key
                    )


                    st.success(
                        "🎉 Workout completed and saved!"
                    )


                except Exception as error:

                    st.error(
                        "Workout could not be saved."
                    )

                    st.code(
                        str(error)
                    )


# ============================================================
# WORKOUT RESULT
# ============================================================

if st.session_state.last_workout_result:

    result = (
        st.session_state.last_workout_result
    )

    st.divider()

    st.subheader(
        "🎉 Workout Result"
    )


    r1, r2, r3 = (
        st.columns(3)
    )


    r1.metric(
        "🔁 Reps",
        result.get(
            "reps",
            0
        )
    )


    r2.metric(
        "⭐ XP Earned",
        result.get(
            "xp_earned",
            0
        )
    )


    r3.metric(
        "🏆 Level",
        result.get(
            "level",
            1
        )
    )


    st.success(
        f"🔥 Current Streak: "
        f"{result.get('streak', 0)} Days"
    )


    st.success(
        f"⭐ Total XP: "
        f"{result.get('total_xp', 0)}"
    )


# ============================================================
# END
# ============================================================