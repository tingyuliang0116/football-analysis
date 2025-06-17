import streamlit as st
import pandas as pd
import numpy as np
from utils.pickle_u import load_detections
import time
import os 

class PlayerStats:
    def __init__(self):
        self.transformed_points = load_detections("pickle/bb_transformed_point.pkl")
        self.detection_data = load_detections("pickle/bb_detection.pkl")
        self.speed_data = load_detections("pickle/bb_speed.pkl")  # Load pre-calculated speed data
        self.distance_data = load_detections("pickle/bb_distance.pkl")  # Load pre-calculated distance data
        self.ball_possession = load_detections("pickle/bb_possession.pkl")  # Load pre-calculated possession data
        self.fps = 60
        self.window = 12
        self.current_frame = 0
        self.team_players = {0: set(), 1: set()}  # Track players by team

    def update_player_teams(self, frame_idx):
        if frame_idx >= len(self.detection_data):
            return
            
        all_detections, _ = self.detection_data[frame_idx]
        player_detections = all_detections[all_detections.class_id != 2]  # Exclude referee
        
        # Update team rosters
        for tracker_id, class_id in zip(player_detections.tracker_id, player_detections.class_id):
            if tracker_id is not None:  # Make sure tracker_id exists
                self.team_players[class_id].add(str(tracker_id))

    def get_player_stats(self, player_id):
        current_second = self.current_frame // self.fps
        str_player_id = str(player_id)  # Convert player_id to string for dictionary lookup
        
        # Get speed and distance for current second, defaulting to 0 if not found
        current_speed = self.speed_data.get(current_second, {}).get(str_player_id, 0)
        total_distance = self.distance_data.get(current_second, {}).get(str_player_id, 0)
        
        return {
            'Current Speed': f"{current_speed:.1f} km/h",
            'Total Distance': f"{total_distance:.2f} m"
        }

def encode_videos():
    """Encode the combined video with h264 codec"""
    game_path = 'output_videos/bb_combined.mp4'
    output_path = 'output_videos/bb_combined_encode.mp4'
    
    # Re-encode combined video with h264 codec
    os.system(f'ffmpeg -i {game_path} -vcodec h264 -acodec aac {output_path} -y')
    
    return output_path

def main():
    st.set_page_config(layout="wide")
    st.title("Dashboard")
    
    # Initialize session state
    if 'player_stats' not in st.session_state:
        st.session_state.player_stats = PlayerStats()
    if 'current_frame' not in st.session_state:
        st.session_state.current_frame = 0
    if 'play_video' not in st.session_state:
        st.session_state.play_video = False
    
    # Video controls
    col_control1, col_control2, col_control3 = st.columns([1,1,2])
    with col_control1:
        if st.button('Play/Pause'):
            st.session_state.play_video = not st.session_state.play_video
    with col_control2:
        if st.button('Reset'):
            st.session_state.current_frame = 0
            st.session_state.play_video = False
    with col_control3:
        frame_slider = st.slider('Frame', 0, 9, st.session_state.current_frame)
        if frame_slider != st.session_state.current_frame:
            st.session_state.current_frame = frame_slider

    # Video display
    st.subheader("Match Video")
    try:
        video_file = open('output_videos/bb_combined_encode.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes, start_time=st.session_state.current_frame)
        video_file.close()
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")
    
    # Update statistics based on current frame
    stats = st.session_state.player_stats
    current_frame = st.session_state.current_frame * 60
    stats.current_frame = current_frame
    stats.update_player_teams(current_frame)
    
    # Player Statistics
    st.subheader("Player Statistics")
    stats_col1, stats_col2 = st.columns(2)
    
    with stats_col1:
        st.markdown("### Team 1")
        team1_stats = []
        for player_id in stats.team_players[0]:
            player_data = stats.get_player_stats(player_id)
            team1_stats.append({
                'Player': f'Player {player_id}',
                **player_data
            })
        if team1_stats:
            st.table(pd.DataFrame(team1_stats))
    
    with stats_col2:
        st.markdown("### Team 2")
        team2_stats = []
        for player_id in stats.team_players[1]:
            player_data = stats.get_player_stats(player_id)
            team2_stats.append({
                'Player': f'Player {player_id}',
                **player_data
            })
        if team2_stats:
            st.table(pd.DataFrame(team2_stats))
    
    # Match Statistics
    st.subheader("Match Statistics")
    match_stats_col1, match_stats_col2 = st.columns(2)
    
    # Get possession from pre-calculated data
    current_second = st.session_state.current_frame
    possession_data = stats.ball_possession.get(current_second, {0: 50, 1: 50})
    team1_possession = possession_data[0]
    team2_possession = possession_data[1]
    
    with match_stats_col1:
        st.metric("Team 1 Possession", f"{team1_possession:.1f}%")
        st.metric("Shots on Target", "8")
        st.metric("Passes Completed", "245")
    
    with match_stats_col2:
        st.metric("Team 2 Possession", f"{team2_possession:.1f}%")
        st.metric("Shots on Target", "6")
        st.metric("Passes Completed", "198")
    
    # Auto-advance frame if playing
    if st.session_state.play_video:
        time.sleep(1)
        if st.session_state.current_frame < 9:
            st.session_state.current_frame += 1
        else:
            st.session_state.play_video = False
        st.rerun()

if __name__ == "__main__":
    # Encode video first time if needed
    if not os.path.exists('output_videos/bb_combined_encode.mp4'):
        encode_videos()
    main()