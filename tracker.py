import supervision as sv
from tqdm import tqdm
import numpy as np
from ultralytics import YOLO
from utils.team import TeamClassifier, resolve_goalkeepers_team_id
from utils.pickle_u import load_detections, save_detections
from utils.ball_assign import assign_ball_to_player
from utils.config_pitch import SoccerPitchConfiguration
from utils.view import ViewTransformer
from utils.draw_pitch import draw_pitch, draw_points_on_pitch

BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3
# PLAYER_DETECTION_MODEL_PATH = ""
# PITCH_DETECTION_MODEL_PATH = ""
# source_video_path = ""
STRIDE = 60
CONFIG = SoccerPitchConfiguration()

# player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH)
# pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH)

ellipse_annotator = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700"]), thickness=2
)
label_annotator = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(["#00BFFF", "#FF1493", "#FFD700"]),
    text_color=sv.Color.from_hex("#000000"),
    text_position=sv.Position.BOTTOM_CENTER,
)
triangle_annotator = sv.TriangleAnnotator(
    color=sv.Color.from_hex("#FFD700"), base=20, height=17
)

# frame_generator = sv.get_video_frames_generator(
#     source_path=source_video_path, stride=STRIDE
# )

# crops = []
# for frame in tqdm(frame_generator, desc="collecting crops"):
#     results = player_detection_model.predict(frame, conf=0.3)[0]
#     detections = sv.Detections.from_ultralytics(results)
#     players_detections = detections[detections.class_id == PLAYER_ID]
#     players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
#     crops += players_crops

# team_classifier = TeamClassifier(device="cuda")
# team_classifier.fit(crops)

tracker = sv.ByteTrack()

detections_data = load_detections("pickle/bb_detection.pkl")
pitch_data = load_detections("pickle/bb_point.pkl")

def callback(frame, index):
    global detections_data, point
    all_detections, ball_detections = detections_data[index]
    key_points = pitch_data[index] 
    # results = player_detection_model.predict(frame, conf=0.3)[0]
    # detections = sv.Detections.from_ultralytics(results)

    # ball_detections = detections[detections.class_id == BALL_ID]
    # ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

    # all_detections = detections[detections.class_id != BALL_ID]
    # all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
    # all_detections = tracker.update_with_detections(detections=all_detections)

    # goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
    # players_detections = all_detections[all_detections.class_id == PLAYER_ID]
    # referees_detections = all_detections[all_detections.class_id == REFEREE_ID]

    # players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
    # players_detections.class_id = team_classifier.predict(players_crops)

    # goalkeepers_detections.class_id = resolve_goalkeepers_team_id(
    #     players_detections, goalkeepers_detections)

    # referees_detections.class_id -= 1

    # all_detections = sv.Detections.merge([
    #     players_detections, goalkeepers_detections, referees_detections])

    labels = [str(tracker_id) for tracker_id in all_detections.tracker_id]
    
    annotated_frame = frame.copy()
    annotated_frame = ellipse_annotator.annotate(
        scene=annotated_frame, detections=all_detections
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=all_detections, labels=labels
    )
    annotated_frame = triangle_annotator.annotate(
        scene=annotated_frame, detections=ball_detections
    )
    
    players_detections = all_detections[all_detections.class_id != 2]
    referees_detections = all_detections[all_detections.class_id == 2]
    
    if ball_detections:
        ball_player = assign_ball_to_player(players_detections, ball_detections)
        annotated_frame = triangle_annotator.annotate(
            scene=annotated_frame, detections=ball_player
        )

    # all_detections.class_id = all_detections.class_id.astype(int)
    # detections_data[index] = (all_detections, ball_detections)
    

    # results = pitch_detection_model.predict(frame, conf=0.3)[0]
    # key_points = sv.KeyPoints.from_inference(results)
    # pitch_data[index] = key_points
    
    filter = key_points.confidence[0] > 0.5
    frame_reference_points = key_points.xy[0][filter]
    pitch_reference_points = np.array(CONFIG.vertices)[filter]

    transformer = ViewTransformer(
        source=frame_reference_points, target=pitch_reference_points
    )

    frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

    players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_players_xy = transformer.transform_points(points=players_xy)
    player_tracker_id = players_detections.tracker_id
    
    referees_xy = referees_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_referees_xy = transformer.transform_points(points=referees_xy)
    radar = draw_pitch(CONFIG)
    radar = draw_points_on_pitch(
        config=CONFIG,
        xy=pitch_ball_xy,
        face_color=sv.Color.WHITE,
        edge_color=sv.Color.BLACK,
        radius=10,
        pitch=radar,
    )
    radar = draw_points_on_pitch(
        config=CONFIG,
        xy=pitch_players_xy[players_detections.class_id == 0],
        face_color=sv.Color.from_hex("00BFFF"),
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=radar,
    )
    radar = draw_points_on_pitch(
        config=CONFIG,
        xy=pitch_players_xy[players_detections.class_id == 1],
        face_color=sv.Color.from_hex("FF1493"),
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=radar,
    )
    radar = draw_points_on_pitch(
        config=CONFIG,
        xy=pitch_referees_xy,
        face_color=sv.Color.from_hex("FFD700"),
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=radar,
    )

    h, w, _ = frame.shape
    # Resize both frames to be the same size
    annotated_frame = sv.resize_image(annotated_frame, (w, h))
    radar = sv.resize_image(radar, (w, h))  # Resize radar to full width, full height
    combined_frame = np.hstack((annotated_frame, radar))  # Stack horizontally
    
    return combined_frame  # Return the combined frame


def process_split_videos(source_path, target_base_path):
    video_info = sv.VideoInfo.from_video_path(source_path)
    
    # Create new video info with doubled width for combined frame
    combined_video_info = sv.VideoInfo(
        width=video_info.width * 2,  # Double width for two equal-sized frames
        height=video_info.height,
        fps=video_info.fps
    )
    
    with sv.VideoSink(target_path=f"{target_base_path}_combined.mp4", video_info=combined_video_info) as frame_sink:
        frame_generator = sv.get_video_frames_generator(source_path)
        
        for index, frame in enumerate(tqdm(frame_generator)):
            result = callback(frame, index)
            frame_sink.write_frame(result)

process_split_videos(
    source_path="input_videos/bb.mov",
    target_base_path="output_videos/bb"
)

# save_detections(detections_data, "pickle/bb_detection.pkl")
# save_detections(pitch_data, "pickle/bb_point.pkl")