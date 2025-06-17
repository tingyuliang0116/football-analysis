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
from utils.distance_speed import calculate_player_speeds
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3
STRIDE = 60
CONFIG = SoccerPitchConfiguration()

tracker = sv.ByteTrack()
detections_data = load_detections("pickle/bb_detection.pkl")
pitch_data = load_detections("pickle/bb_point.pkl")
transformed_point = {}
player_stats = {} 
team_players = {0: set(), 1: set()}
team1_ball_possession = 0
team2_ball_possession = 0
ball_possession = {}
def process_track(index):
    global detections_data, pitch_data, player_stats, team_players, team1_ball_possession, team2_ball_possession
    all_detections, ball_detections = detections_data[index]
    key_points = pitch_data[index]

    players_detections = all_detections[all_detections.class_id != 2]
    for tracker_id, class_id in zip(players_detections.tracker_id, players_detections.class_id):
        if tracker_id is not None:  
            team_players[class_id].add(str(tracker_id))
            
    if ball_detections:
        ball_player = assign_ball_to_player(players_detections, ball_detections)
        for team_id, players in team_players.items():
            if str(ball_player.tracker_id[0]) in players:
                if team_id == 0:
                    team1_ball_possession += 1
                elif team_id == 1:
                    team2_ball_possession += 1
                break
        team1 = team1_ball_possession / (team1_ball_possession + team2_ball_possession) *100
        team2 = team2_ball_possession / (team1_ball_possession + team2_ball_possession) *100
        ball_possession[index] = {0:team1, 1:team2}

    filter = key_points.confidence[0] > 0.5
    frame_reference_points = key_points.xy[0][filter]
    pitch_reference_points = np.array(CONFIG.vertices)[filter]

    transformer = ViewTransformer(
        source=frame_reference_points, target=pitch_reference_points
    )

    players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_players_xy = transformer.transform_points(points=players_xy)
    player_tracker_id = players_detections.tracker_id
    transformed_point[index] = {tracker_id: list(coord) for tracker_id, coord in zip(player_tracker_id, pitch_players_xy)}

    return 

video_info = sv.VideoInfo.from_video_path("output_videos/bb_game_encode.mp4")
for i in range(video_info.total_frames):
    process_track(i)
ball_possession = {int(k/60): v for k, v in ball_possession.items() if k % 60 == 0}
speed, distance = calculate_player_speeds(transformed_point, video_info.fps, video_info.total_frames)
# save_detections(speed, "pickle/bb_speed.pkl")
# save_detections(distance, "pickle/bb_distance.pkl")
# save_detections(ball_possession, "pickle/bb_possession.pkl")
speed_data = load_detections("pickle/bb_speed.pkl")
print(speed_data)