import supervision as sv
def measure_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def assign_ball_to_player(player_detections, ball_detections):
    ball = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)[0]
    min_distance = 99999
    assigned_player = player_detections[0]
    for index, player in enumerate(player_detections.xyxy):
        player_bbox = player
        distance_left = measure_distance((player_bbox[0],player_bbox[-1]),ball)
        distance_right = measure_distance((player_bbox[2],player_bbox[-1]),ball)
        distance = min(distance_left, distance_right)
        if distance < min_distance:
            min_distance = distance
            assigned_player = player_detections[index]
    return assigned_player