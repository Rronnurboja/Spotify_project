
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Spotify Mood Mixer", layout="wide", page_icon="🎵")
st.title("🎵 Spotify Mood Mixer")
st.caption("Find similar songs and create perfect playlists")

# ========== LOAD DATA ==========
@st.cache_data
def load_data():
    df = pd.read_csv("data/SpotifyFeatures.csv")
    df = df.drop_duplicates(subset='track_id').reset_index(drop=True)
    df['duration_min'] = df['duration_ms'] / 60000
    return df

df = load_data()

# ========== SIDEBAR CONTROLS ==========
st.sidebar.header("🎛️ Controls")

# Mood selector
mood_options = {
    "😊 Happy & Energetic": {"target": 0.8, "range": 0.2},
    "😢 Sad & Melancholic": {"target": 0.3, "range": 0.2},
    "💃 Dance Party": {"target": 0.8, "range": 0.2},
    "🧘 Calm & Relaxed": {"target": 0.3, "range": 0.2},
    "🎸 Rock Out": {"target": 0.7, "range": 0.3},
    "✨ Custom Mood": {"target": 0.5, "range": 0.3}
}

selected_mood = st.sidebar.selectbox(
    "Mood",
    list(mood_options.keys()),
    index=0
)

# Feature sliders (single slider per feature with range)
st.sidebar.subheader("🎚️ Audio Features")

if selected_mood == "✨ Custom Mood":
    dance_target = st.sidebar.slider(
        "Danceability Target", 0.0, 1.0, 0.5, 0.1,
        help="Center of the range"
    )
    dance_range = st.sidebar.slider(
        "Danceability Range", 0.1, 0.5, 0.3, 0.05,
        help="How wide to search (± this value)"
    )
    
    energy_target = st.sidebar.slider(
        "Energy Target", 0.0, 1.0, 0.5, 0.1
    )
    energy_range = st.sidebar.slider(
        "Energy Range", 0.1, 0.5, 0.3, 0.05
    )
    
    valence_target = st.sidebar.slider(
        "Happiness Target", 0.0, 1.0, 0.5, 0.1
    )
    valence_range = st.sidebar.slider(
        "Happiness Range", 0.1, 0.5, 0.3, 0.05
    )
else:
    # Use preset values
    preset = mood_options[selected_mood]
    dance_target = preset["target"]
    dance_range = preset["range"]
    energy_target = preset["target"]
    energy_range = preset["range"]
    valence_target = preset["target"]
    valence_range = preset["range"]
    
    # Show preset values
    st.sidebar.write(f"Danceability Target: **{dance_target}** ± {dance_range}")

# Tempo range (single slider)
tempo_target = st.sidebar.slider(
    "Tempo Target (BPM)", 60, 200, 120, 10,
    help="Center tempo"
)
tempo_range = st.sidebar.slider(
    "Tempo Range", 20, 80, 40, 5,
    help="Search range (± BPM)"
)

# Genre filter
st.sidebar.subheader("🎵 Genres")
all_genres = sorted(df['genre'].unique())
selected_genres = st.sidebar.multiselect(
    "Filter genres",
    all_genres,
    default=['Pop', 'Rock'] if 'Pop' in all_genres else []
)

# Advanced options
with st.sidebar.expander("⚙️ Advanced Options"):
    min_popularity = st.slider("Minimum Popularity", 0, 100, 40)
    max_duration = st.slider("Max Song Length (min)", 1.0, 10.0, 6.0, 0.5)
    similarity_method = st.selectbox(
        "Similarity Method",
        ["Smart (genre-aware)", "Simple (all features)"]
    )

# ========== MAIN TABS ==========
tab1, tab2, tab3 = st.tabs(["🔍 Smart Search", "🎧 Create Playlist", "📊 Explore Data"])

with tab1:
    st.header("Find Similar Songs")
    
    # Search box
    col1, col2, col3 = st.columns([3, 2, 1])
    with col1:
        song_query = st.text_input("Song", "Shape of You", 
                                 help="Enter song name or part of it")
    with col2:
        artist_query = st.text_input("Artist", "Ed Sheeran")
    with col3:
        st.write("")  # Spacer
        st.write("")
        search_btn = st.button("🔍 Search", type="primary", use_container_width=True)
    
    if search_btn and song_query:
        with st.spinner("Finding similar songs..."):
            # Find target song
            mask = (df['track_name'].str.contains(song_query, case=False, na=False)) & \
                   (df['artist_name'].str.contains(artist_query, case=False, na=False))
            
            if mask.any():
                target = df[mask].iloc[0]
                
                # Display target
                st.success(f"🎵 **{target['track_name']}** by **{target['artist_name']}**")
                
                # Target info in nice cards
                cols = st.columns(5)
                cols[0].metric("Genre", target['genre'])
                cols[1].metric("Dance", f"{target['danceability']:.2f}")
                cols[2].metric("Energy", f"{target['energy']:.2f}")
                cols[3].metric("Happy", f"{target['valence']:.2f}")
                cols[4].metric("Popularity", f"{target['popularity']}/100")
                
                # Filter dataset
                search_df = df.copy()
                
                # Apply genre filter
                if selected_genres:
                    search_df = search_df[search_df['genre'].isin(selected_genres)]
                
                # Apply feature filters
                search_df = search_df[
                    (search_df['danceability'] >= dance_target - dance_range) &
                    (search_df['danceability'] <= dance_target + dance_range) &
                    (search_df['energy'] >= energy_target - energy_range) &
                    (search_df['energy'] <= energy_target + energy_range) &
                    (search_df['valence'] >= valence_target - valence_range) &
                    (search_df['valence'] <= valence_target + valence_range) &
                    (search_df['tempo'] >= tempo_target - tempo_range) &
                    (search_df['tempo'] <= tempo_target + tempo_range) &
                    (search_df['popularity'] >= min_popularity) &
                    (search_df['duration_min'] <= max_duration)
                ]
                
                # Remove target
                search_df = search_df[search_df['track_id'] != target['track_id']]
                
                # Calculate similarity
                if similarity_method == "Smart (genre-aware)":
                    # Weight features based on genre
                    if target['genre'] in ['Pop', 'Rock', 'Electronic']:
                        weights = np.array([0.4, 0.3, 0.3])  # dance, energy, valence
                    elif target['genre'] in ['Latin', 'Reggaeton']:
                        weights = np.array([0.2, 0.4, 0.4])  # dance, energy, tempo
                    else:
                        weights = np.array([0.33, 0.33, 0.34])
                    
                    features = ['danceability', 'energy', 'valence']
                    target_features = target[features].values
                    
                    similarities = []
                    for _, song in search_df.iterrows():
                        song_features = song[features].values
                        diff = np.abs(song_features - target_features)
                        weighted_diff = np.sum(diff * weights)
                        similarity = 1 - weighted_diff
                        similarities.append(similarity)
                    
                    search_df['similarity'] = similarities
                else:
                    # Simple Euclidean distance
                    features = ['danceability', 'energy', 'valence', 'tempo']
                    target_features = target[features].values
                    
                    def calc_similarity(row):
                        song_features = row[features].values
                        distance = np.sqrt(np.sum((song_features - target_features) ** 2))
                        return 1 / (1 + distance)
                    
                    search_df['similarity'] = search_df.apply(calc_similarity, axis=1)
                
                # Get top recommendations
                recommendations = search_df.nlargest(10, 'similarity')
                
                if len(recommendations) > 0:
                    st.subheader(f"🎧 Top {len(recommendations)} Recommendations")
                    
                    # Display as nice cards
                    for i, (_, song) in enumerate(recommendations.iterrows(), 1):
                        with st.container():
                            col1, col2, col3, col4 = st.columns([4, 2, 2, 1])
                            with col1:
                                st.write(f"**{i}. {song['track_name'][:35]}**")
                                st.caption(f"👤 {song['artist_name']} | 🎵 {song['genre']}")
                            with col2:
                                st.write(f"💃 {song['danceability']:.2f}")
                                st.write(f"⚡ {song['energy']:.2f}")
                            with col3:
                                st.write(f"😊 {song['valence']:.2f}")
                                st.write(f"⭐ {song['popularity']}/100")
                            with col4:
                                st.metric("Match", f"{song['similarity']:.2f}")
                            st.divider()
                else:
                    st.warning("No similar songs found. Try relaxing filters.")
            else:
                st.error("Song not found! Try a different search.")

with tab2:
    st.header("Create Playlist")
    
    # ========== NEW: PLAYLIST CREATION OPTIONS ==========
    playlist_mode = st.radio(
        "Create playlist from:",
        ["🎛️ Mood & Features", "👤 Specific Artist", "🔤 Similar Song Names"],
        horizontal=True
    )
    
    if playlist_mode == "🎛️ Mood & Features":
        # Original mood-based playlist creation
        col1, col2 = st.columns(2)
        with col1:
            playlist_size = st.slider("Number of songs", 5, 50, 15)
        with col2:
            playlist_name = st.text_input("Playlist name", "My Mood Playlist")
    
    elif playlist_mode == "👤 Specific Artist":
        # NEW: Artist-based playlist
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            artist_for_playlist = st.text_input("Artist name", "Ed Sheeran", 
                                              help="Create playlist from this artist's style")
        with col2:
            playlist_size = st.slider("Number of songs", 5, 50, 15)
        with col3:
            include_artist = st.checkbox("Include artist", True, 
                                       help="Include songs by this artist in the playlist")
        
        playlist_name = st.text_input("Playlist name", f"Songs like {artist_for_playlist}")
        
        # Additional options for artist-based playlist
        with st.expander("🎨 Artist Style Options"):
            style_match = st.slider("Style Match Strength", 0.1, 1.0, 0.7, 0.1,
                                  help="How closely to match the artist's typical style")
            include_collabs = st.checkbox("Include collaborations", True,
                                        help="Include songs where artist appears with others")
    
    elif playlist_mode == "🔤 Similar Song Names":
        # NEW: Song name similarity playlist
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            song_name_pattern = st.text_input("Song name pattern", "Love", 
                                            help="Find songs with similar names (e.g., 'Love', 'Night', 'Dream')")
        with col2:
            playlist_size = st.slider("Number of songs", 5, 50, 15)
        with col3:
            exact_match = st.checkbox("Exact words", False,
                                    help="Match exact words only (not partial matches)")
        
        playlist_name = st.text_input("Playlist name", f"Songs about {song_name_pattern}")
        
        # Additional options for name-based playlist
        with st.expander("📝 Name Matching Options"):
            match_strength = st.select_slider(
                "Match strength",
                options=["Loose", "Medium", "Strict"],
                value="Medium"
            )
    
    # ========== GENERATE PLAYLIST BUTTON ==========
    if st.button("🎵 Generate Playlist", type="primary", use_container_width=True):
        with st.spinner("Creating your playlist..."):
            
            if playlist_mode == "🎛️ Mood & Features":
                # Original mood-based filtering
                playlist_df = df.copy()
                
                if selected_genres:
                    playlist_df = playlist_df[playlist_df['genre'].isin(selected_genres)]
                
                playlist_df = playlist_df[
                    (playlist_df['danceability'] >= dance_target - dance_range) &
                    (playlist_df['danceability'] <= dance_target + dance_range) &
                    (playlist_df['energy'] >= energy_target - energy_range) &
                    (playlist_df['energy'] <= energy_target + energy_range) &
                    (playlist_df['valence'] >= valence_target - valence_range) &
                    (playlist_df['valence'] <= valence_target + valence_range) &
                    (playlist_df['tempo'] >= tempo_target - tempo_range) &
                    (playlist_df['tempo'] <= tempo_target + tempo_range) &
                    (playlist_df['popularity'] >= min_popularity) &
                    (playlist_df['duration_min'] <= max_duration)
                ]
                
                # Create balanced playlist
                if len(playlist_df) >= playlist_size:
                    playlist = playlist_df.sample(playlist_size, random_state=42)
                else:
                    playlist = playlist_df
                
                st.success(f"✅ **{playlist_name}** created with {len(playlist)} songs!")
            
            elif playlist_mode == "👤 Specific Artist":
                # NEW: Find artist and similar songs
                artist_mask = df['artist_name'].str.contains(artist_for_playlist, case=False, na=False)
                
                if not artist_mask.any():
                    st.error(f"Artist '{artist_for_playlist}' not found in dataset!")
                    st.stop()
                
                # Get artist's songs
                artist_songs = df[artist_mask]
                
                if len(artist_songs) == 0:
                    st.error(f"No songs found for artist '{artist_for_playlist}'!")
                    st.stop()
                
                # Calculate average features of this artist
                features = ['danceability', 'energy', 'valence', 'tempo', 'acousticness']
                artist_avg_features = artist_songs[features].mean()
                
                # Find songs with similar features
                all_songs = df.copy()
                
                # Calculate similarity to artist's average style
                def artist_similarity(row):
                    song_features = row[features].values
                    artist_features = artist_avg_features.values
                    distance = np.sqrt(np.sum((song_features - artist_features) ** 2))
                    return 1 / (1 + distance)
                
                all_songs['artist_similarity'] = all_songs.apply(artist_similarity, axis=1)
                
                # Filter by similarity threshold
                similarity_threshold = 1.0 - style_match  # Convert slider to threshold
                similar_songs = all_songs[all_songs['artist_similarity'] >= similarity_threshold]
                
                # Include/exclude artist's own songs
                if not include_artist:
                    similar_songs = similar_songs[~similar_songs['artist_name'].str.contains(artist_for_playlist, case=False)]
                
                # Include/exclude collaborations
                if not include_collabs:
                    # Remove songs where artist appears with others (contains 'feat.', '&', 'with', etc.)
                    collab_pattern = r'(feat\.|ft\.|featuring|&|with|x|,)'
                    similar_songs = similar_songs[~similar_songs['artist_name'].str.contains(collab_pattern, case=False, na=False)]
                
                # Create playlist
                if len(similar_songs) >= playlist_size:
                    playlist = similar_songs.nlargest(playlist_size, 'artist_similarity')
                else:
                    playlist = similar_songs
                
                # Add some of artist's own songs if requested
                if include_artist and len(artist_songs) > 0:
                    # Take top popular songs by the artist
                    artist_top = artist_songs.nlargest(min(3, len(artist_songs)), 'popularity')
                    playlist = pd.concat([artist_top, playlist]).drop_duplicates(subset='track_id').head(playlist_size)
                
                st.success(f"✅ **{playlist_name}** created with {len(playlist)} songs in style of {artist_for_playlist}!")
            
            elif playlist_mode == "🔤 Similar Song Names":
                # NEW: Find songs with similar names
                if match_strength == "Strict":
                    # Exact word matching
                    pattern = r'\b' + song_name_pattern + r'\b'
                    name_mask = df['track_name'].str.contains(pattern, case=False, na=False, regex=True)
                elif match_strength == "Medium":
                    # Match whole word, but not as strict
                    name_mask = df['track_name'].str.contains(r'\b' + song_name_pattern + r'\w*', case=False, na=False, regex=True)
                else:  # Loose
                    # Partial matching anywhere in the name
                    name_mask = df['track_name'].str.contains(song_name_pattern, case=False, na=False)
                
                name_songs = df[name_mask]
                
                if len(name_songs) == 0:
                    st.error(f"No songs found with name containing '{song_name_pattern}'!")
                    st.stop()
                
                # Also apply mood/filters if user wants
                apply_filters = st.checkbox("Apply mood/filters to name matches", True)
                
                if apply_filters:
                    # Apply the same filters as mood-based playlist
                    filtered_songs = name_songs.copy()
                    
                    if selected_genres:
                        filtered_songs = filtered_songs[filtered_songs['genre'].isin(selected_genres)]
                    
                    filtered_songs = filtered_songs[
                        (filtered_songs['danceability'] >= dance_target - dance_range) &
                        (filtered_songs['danceability'] <= dance_target + dance_range) &
                        (filtered_songs['energy'] >= energy_target - energy_range) &
                        (filtered_songs['energy'] <= energy_target + energy_range) &
                        (filtered_songs['valence'] >= valence_target - valence_range) &
                        (filtered_songs['valence'] <= valence_target + valence_range) &
                        (filtered_songs['tempo'] >= tempo_target - tempo_range) &
                        (filtered_songs['tempo'] <= tempo_target + tempo_range) &
                        (filtered_songs['popularity'] >= min_popularity) &
                        (filtered_songs['duration_min'] <= max_duration)
                    ]
                    
                    if len(filtered_songs) >= playlist_size:
                        playlist = filtered_songs.sample(playlist_size, random_state=42)
                    else:
                        playlist = filtered_songs
                else:
                    # Just take most popular songs with matching names
                    playlist = name_songs.nlargest(playlist_size, 'popularity')
                
                st.success(f"✅ **{playlist_name}** created with {len(playlist)} songs about '{song_name_pattern}'!")
            
            # ========== COMMON PLAYLIST DISPLAY (ALL MODES) ==========
            # Sort by energy for good flow
            playlist = playlist.sort_values('energy')
            
            # Display playlist stats
            total_minutes = playlist['duration_min'].sum()
            avg_popularity = playlist['popularity'].mean()
            
            cols = st.columns(4)
            cols[0].metric("Total Duration", f"{total_minutes:.1f} min")
            cols[1].metric("Avg Popularity", f"{avg_popularity:.1f}/100")
            cols[2].metric("Genres", playlist['genre'].nunique())
            cols[3].metric("Artists", playlist['artist_name'].nunique())
            
            # Playlist table
            st.subheader("🎶 Playlist")
            playlist_display = playlist[['track_name', 'artist_name', 'genre', 'popularity', 'duration_min']].copy()
            playlist_display['duration'] = playlist_display['duration_min'].apply(lambda x: f"{x:.1f} min")
            playlist_display = playlist_display.rename(columns={
                'track_name': 'Song Name',
                'artist_name': 'Artist Name',
                'genre': 'Genre',
                'popularity': 'Popularity'
            }).drop(columns=['duration_min'])
            
            # Add row numbers
            playlist_display.index = range(1, len(playlist_display) + 1)
            
            # Style the dataframe
            st.dataframe(
                playlist_display,
                use_container_width=True,
                column_config={
                    "Song Name": st.column_config.TextColumn(
                        "Song Name",
                        width="large",
                        help="Name of the song"
                    ),
                    "Artist Name": st.column_config.TextColumn(
                        "Artist Name",
                        width="medium",
                        help="Name of the artist"
                    ),
                    "Genre": st.column_config.TextColumn(
                        "Genre",
                        width="small",
                        help="Music genre"
                    ),
                    "Popularity": st.column_config.ProgressColumn(
                        "Popularity",
                        min_value=0,
                        max_value=100,
                        format="%d",
                        help="Popularity score from 0-100"
                    ),
                    "duration": st.column_config.TextColumn(
                        "Duration",
                        help="Song length in minutes"
                    )
                }
            )
            
            # List view
            st.subheader("📋 Playlist List View")
            for i, (_, song) in enumerate(playlist.iterrows(), 1):
                duration_min = song['duration_ms'] / 60000
                with st.container():
                    col1, col2, col3 = st.columns([4, 2, 2])
                    with col1:
                        st.write(f"**{i}. {song['track_name']}**")
                        st.caption(f"by **{song['artist_name']}**")
                    with col2:
                        st.write(f"🎵 {song['genre']}")
                        st.write(f"⏱️ {duration_min:.1f} min")
                    with col3:
                        st.write(f"⭐ {song['popularity']}/100")
                        st.write(f"💃 {song['danceability']:.2f}")
                    st.divider()
            
            # 📥 CSV Download Button (same for all modes)
            st.subheader("💾 Export Playlist")
            
            csv_data = playlist[['track_name', 'artist_name', 'genre', 'popularity', 
                               'danceability', 'energy', 'valence', 'tempo', 'duration_ms']].copy()
            csv_data['duration_min'] = csv_data['duration_ms'] / 60000
            csv_data = csv_data.drop(columns=['duration_ms'])
            csv_data = csv_data.rename(columns={
                'track_name': 'Song_Name',
                'artist_name': 'Artist_Name',
                'genre': 'Genre',
                'popularity': 'Popularity',
                'danceability': 'Danceability',
                'energy': 'Energy',
                'valence': 'Happiness',
                'tempo': 'Tempo_BPM',
                'duration_min': 'Duration_Min'
            })
            
            csv = csv_data.to_csv(index=False)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"{playlist_name.replace(' ', '_')}.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )
            with col2:
                spotify_text = ""
                for i, (_, song) in enumerate(playlist.iterrows(), 1):
                    spotify_text += f"{i}. {song['track_name']} - {song['artist_name']}\n"
                
                st.download_button(
                    label="📝 Download as Text",
                    data=spotify_text,
                    file_name=f"{playlist_name.replace(' ', '_')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with col3:
                json_data = playlist[['track_name', 'artist_name', 'genre', 'popularity']].to_json(orient='records', indent=2)
                st.download_button(
                    label="📊 Download as JSON",
                    data=json_data,
                    file_name=f"{playlist_name.replace(' ', '_')}.json",
                    mime="application/json",
                    use_container_width=True
                )

            # Show what the CSV contains
            with st.expander("📋 What's in the CSV file?", expanded=False):
                st.write("The CSV file contains these columns:")
                st.write("- **Song_Name**: Name of the song")
                st.write("- **Artist_Name**: Name of the artist")
                st.write("- **Genre**: Music genre (Pop, Rock, etc.)")
                st.write("- **Popularity**: Popularity score from 0-100")
                st.write("- **Danceability**: How danceable the song is (0-1)")
                st.write("- **Energy**: How energetic the song is (0-1)")
                st.write("- **Happiness**: How happy/positive the song is (0-1)")
                st.write("- **Tempo_BPM**: Speed of the song in beats per minute")
                st.write("- **Duration_Min**: Length of the song in minutes")

            # Show audio features summary
            with st.expander("📊 Audio Features Summary", expanded=False):
                features = ['danceability', 'energy', 'valence', 'tempo']
                fig, axes = plt.subplots(2, 2, figsize=(10, 6))
                
                for idx, feature in enumerate(features):
                    ax = axes[idx // 2, idx % 2]
                    ax.hist(playlist[feature], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
                    ax.axvline(playlist[feature].mean(), color='red', linestyle='--', label=f'Mean: {playlist[feature].mean():.2f}')
                    ax.set_title(feature.capitalize())
                    ax.set_xlabel("Value")
                    ax.legend()
                
                plt.tight_layout()
                st.pyplot(fig)
                
with tab3:
    st.header("📊 Explore Your Dataset")
    
    # Stats cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Songs", f"{len(df):,}")
    col2.metric("Unique Artists", df['artist_name'].nunique())
    col3.metric("Genres", df['genre'].nunique())
    col4.metric("Avg Popularity", f"{df['popularity'].mean():.1f}/100")
    
    # Genre distribution
    st.subheader("🎵 Genre Distribution")
    top_genres = df['genre'].value_counts().head(15)
    
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    top_genres.plot(kind='barh', ax=ax1, color='lightblue')
    ax1.set_xlabel("Number of Songs")
    ax1.set_title("Top 15 Genres")
    plt.tight_layout()
    st.pyplot(fig1)
    
    # Feature distributions
    st.subheader("📈 Audio Features")
    feature = st.selectbox("Select feature to explore", 
                          ['danceability', 'energy', 'valence', 'tempo', 'acousticness'])
    
    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    ax2.hist(df[feature].dropna(), bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_title(f"Distribution of {feature}")
    ax2.set_xlabel(feature.capitalize())
    ax2.set_ylabel("Count")
    
    # Box plot by top 5 genres
    top_5_genres = df['genre'].value_counts().head(5).index
    genre_data = [df[df['genre'] == genre][feature].dropna() for genre in top_5_genres]
    
    ax3.boxplot(genre_data, labels=top_5_genres)
    ax3.set_title(f"{feature} by Top Genres")
    ax3.set_xticklabels(top_5_genres, rotation=45, ha='right')
    ax3.set_ylabel(feature.capitalize())
    
    plt.tight_layout()
    st.pyplot(fig2)
    
    # Sample data
    st.subheader("🎧 Sample Songs")
    
    sample_filter = st.selectbox("Filter sample", 
                                ["All", "High Popularity (>80)", "Low Popularity (<20)", 
                                 "High Danceability (>0.8)", "Specific Genre"])
    
    sample_df = df.copy()
    if sample_filter == "High Popularity (>80)":
        sample_df = sample_df[sample_df['popularity'] > 80]
        st.caption(f"Showing {len(sample_df)} highly popular songs")
    elif sample_filter == "Low Popularity (<20)":
        sample_df = sample_df[sample_df['popularity'] < 20]
        st.caption(f"Showing {len(sample_df)} less popular songs")
    elif sample_filter == "High Danceability (>0.8)":
        sample_df = sample_df[sample_df['danceability'] > 0.8]
        st.caption(f"Showing {len(sample_df)} highly danceable songs")
    elif sample_filter == "Specific Genre":
        specific_genre = st.selectbox("Select genre", sorted(df['genre'].unique()))
        sample_df = sample_df[sample_df['genre'] == specific_genre]
        st.caption(f"Showing {len(sample_df)} {specific_genre} songs")
    
    # Display sample
    if len(sample_df) > 0:
        sample = sample_df.sample(min(10, len(sample_df)))[['track_name', 'artist_name', 'genre', 'popularity']]
        sample = sample.sort_values('popularity', ascending=False)
        sample.index = range(1, len(sample) + 1)
        
        st.dataframe(
            sample,
            use_container_width=True,
            column_config={
                "track_name": st.column_config.TextColumn("Song", width="large"),
                "artist_name": st.column_config.TextColumn("Artist"),
                "genre": st.column_config.TextColumn("Genre"),
                "popularity": st.column_config.ProgressColumn(
                    "Popularity",
                    min_value=0,
                    max_value=100,
                    format="%d"
                )
            }
        )
    else:
        st.warning("No songs match the selected filter.")

# Footer
st.divider()
st.caption("🎵 **Spotify Mood Mixer** • 232,725 songs • Created with ❤️ by Rron Nurboja")