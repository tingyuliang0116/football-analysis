import numpy as np
player_matrics = {}
distance_matrics = {}
speed_matrics = {}
def calculate_player_speeds(player_positions, fps, total_frames):
    global player_matrics, distance_matrics, speed_matrics
    speed_frame = {}
    for pid in player_positions[0].keys():
        player_matrics[pid] = 0
        speed_frame[pid] = 0
    distance_matrics[0] = player_matrics.copy()
    speed_matrics[0] = speed_frame
    for index in range(fps, total_frames, fps):
        last_frame = index - fps
        if last_frame not in player_positions or index not in player_positions:
            continue
        current_frame_data = player_positions[index]
        last_frame_data = player_positions[last_frame]
        speed_frame = {}
        for pid, pos in current_frame_data.items():
            if pid not in player_matrics:
                player_matrics[pid] = 0
            if pid not in last_frame_data:
                speed_frame[pid] = 0
                continue
            start_pos = np.array(last_frame_data[pid])
            end_pos = np.array(current_frame_data[pid])
            distance = np.linalg.norm(end_pos - start_pos) * 0.01
            speed = distance * 3.6 
            player_matrics[pid] += distance
            speed_frame[pid] = speed
        distance_matrics[int(index/fps)] = player_matrics.copy()
        speed_matrics[int(index/fps)] = speed_frame
    return speed_matrics, distance_matrics