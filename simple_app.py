
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Spotify Mood Mixer", layout="wide")
st.title("🎵 Spotify Mood Mixer")
st.caption("Build playlists with smart music recommendations")

# Load data with caching
@st.cache_data
def load_data():
    df = pd.read_csv("data/SpotifyFeatures.csv")
    df = df.drop_duplicates(subset='track_id').reset_index(drop=True)
    
    # Convert duration to minutes
    df['duration_min'] = df['duration_ms'] / 60000
    
    return df

df = load_data()

# ========== SIDEBAR: USER PREFERENCES ==========
st.sidebar.header("🎛️ Mood & Preferences")

# Mood selection
mood_options = {
    "😊 Happy & Energetic": {"valence": (0.7, 1.0), "energy": (0.7, 1.0)},
    "😢 Sad & Melancholic": {"valence": (0.0, 0.4), "energy": (0.0, 0.4)},
    "💃 Dance Party": {"danceability": (0.7, 1.0), "energy": (0.6, 1.0)},
    "🧘 Calm & Relaxed": {"energy": (0.0, 0.4), "acousticness": (0.5, 1.0)},
    "😠 Angry & Intense": {"energy": (0.7, 1.0), "valence": (0.0, 0.4)},
    "⚡ Any Mood": {}  # No filters
}

selected_mood = st.sidebar.selectbox(
    "Select Mood",
    list(mood_options.keys()),
    index=0
)

# Genre filter (with your actual genres)
all_genres = sorted(df['genre'].unique())
selected_genres = st.sidebar.multiselect(
    "Filter by Genre",
    all_genres,
    default=['Pop', 'Rock', 'Electronic'] if 'Pop' in all_genres else []
)

# Feature sliders
st.sidebar.header("🎚️ Audio Features")

col1, col2 = st.sidebar.columns(2)
with col1:
    min_dance = st.slider("Min Danceability", 0.0, 0.5, 0.1, 0.1)
    min_energy = st.slider("Min Energy", 0.0, 0.5, 0.1, 0.1)
    min_valence = st.slider("Min Happiness", 0.0, 0.5, 0.1, 0.1)
    
with col2:
    max_dance = st.slider("Max Danceability", 0.5, 1.0, 1.0, 0.1)
    max_energy = st.slider("Max Energy", 0.5, 1.0, 1.0, 0.1)
    max_valence = st.slider("Max Happiness", 0.5, 1.0, 1.0, 0.1)

# Tempo range
tempo_range = st.sidebar.slider(
    "Tempo Range (BPM)",
    60, 200, (80, 160),
    help="Slow (60-100) | Medium (100-140) | Fast (140-200)"
)

# Duration filter
duration_range = st.sidebar.slider(
    "Song Length (minutes)",
    1.0, 10.0, (2.0, 5.0),
    0.5
)

# ========== MAIN AREA ==========
tab1, tab2, tab3 = st.tabs(["🔍 Smart Search", "🎧 Generate Playlist", "📊 Dataset Explorer"])

with tab1:
    st.header("Smart Song Search")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        song_query = st.text_input("Song name", "Shape of You", 
                                 help="Enter part of the song title")
    with col2:
        artist_query = st.text_input("Artist name", "Ed Sheeran",
                                   help="Enter part of the artist name")
    
    # SMART SEARCH button
    if st.button("🎯 Find Smart Recommendations", type="primary"):
        with st.spinner("Finding musically similar songs..."):
            # Filter dataset first by user preferences
            filtered_df = df.copy()
            
            # Apply genre filter
            if selected_genres:
                filtered_df = filtered_df[filtered_df['genre'].isin(selected_genres)]
            
            # Apply mood filter
            mood_filters = mood_options[selected_mood]
            for feature, (min_val, max_val) in mood_filters.items():
                if feature in filtered_df.columns:
                    filtered_df = filtered_df[
                        (filtered_df[feature] >= min_val) & 
                        (filtered_df[feature] <= max_val)
                    ]
            
            # Apply feature sliders
            filtered_df = filtered_df[
                (filtered_df['danceability'] >= min_dance) &
                (filtered_df['danceability'] <= max_dance) &
                (filtered_df['energy'] >= min_energy) &
                (filtered_df['energy'] <= max_energy) &
                (filtered_df['valence'] >= min_valence) &
                (filtered_df['valence'] <= max_valence) &
                (filtered_df['tempo'] >= tempo_range[0]) &
                (filtered_df['tempo'] <= tempo_range[1]) &
                (filtered_df['duration_min'] >= duration_range[0]) &
                (filtered_df['duration_min'] <= duration_range[1])
            ]
            
            # Find target song
            mask = (df['track_name'].str.contains(song_query, case=False, na=False)) & \
                   (df['artist_name'].str.contains(artist_query, case=False, na=False))
            
            if mask.any():
                target = df[mask].iloc[0]
                
                # Display target song
                st.success(f"🎵 Found: **{target['track_name']}** by **{target['artist_name']}**")
                
                # Song info in columns
                info_cols = st.columns(5)
                info_cols[0].metric("Genre", target['genre'])
                info_cols[1].metric("Dance", f"{target['danceability']:.2f}")
                info_cols[2].metric("Energy", f"{target['energy']:.2f}")
                info_cols[3].metric("Happy", f"{target['valence']:.2f}")
                info_cols[4].metric("Tempo", f"{target['tempo']:.0f}")
                
                # SMART SIMILARITY: Weight features based on genre context
                if target['genre'] in ['Pop', 'Rock', 'Electronic', 'Hip-Hop', 'R&B']:
                    # Western music: focus on danceability, energy, valence
                    feature_weights = {'danceability': 0.4, 'energy': 0.3, 'valence': 0.3}
                    features = ['danceability', 'energy', 'valence']
                elif target['genre'] in ['Latin', 'Reggaeton', 'Salsa', 'Reggae']:
                    # Latin music: focus more on tempo and energy
                    feature_weights = {'tempo': 0.4, 'energy': 0.4, 'danceability': 0.2}
                    features = ['tempo', 'energy', 'danceability']
                elif target['genre'] in ['Classical', 'Jazz', 'Blues']:
                    # Classical/Jazz: focus on acousticness and tempo
                    feature_weights = {'acousticness': 0.4, 'tempo': 0.3, 'energy': 0.3}
                    features = ['acousticness', 'tempo', 'energy']
                elif target['genre'] == 'Movie':
                    # Movie soundtracks: focus on energy and valence (emotional)
                    feature_weights = {'energy': 0.4, 'valence': 0.4, 'danceability': 0.2}
                    features = ['energy', 'valence', 'danceability']
                else:
                    # Default weights
                    feature_weights = {'danceability': 0.3, 'energy': 0.3, 'valence': 0.2, 'tempo': 0.2}
                    features = ['danceability', 'energy', 'valence', 'tempo']
                
                # Calculate weighted similarity
                similarities = []
                for idx, song in filtered_df.iterrows():
                    if song['track_id'] == target['track_id']:
                        continue  # Skip target
                    
                    similarity = 0
                    for feature in features:
                        weight = feature_weights.get(feature, 0.1)
                        # Normalized difference (0 = identical, 1 = completely different)
                        if feature == 'tempo':
                            diff = abs(song[feature] - target[feature]) / 140  # Normalize by typical range
                        elif feature == 'loudness':
                            diff = abs(song[feature] - target[feature]) / 60   # Loudness range
                        else:
                            diff = abs(song[feature] - target[feature])
                        similarity += (1 - diff) * weight
                    
                    # Bonus for same genre
                    if song['genre'] == target['genre']:
                        similarity += 0.1
                    
                    similarities.append((idx, similarity))
                
                # Sort by similarity
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                # Get top 10 recommendations
                top_indices = [idx for idx, _ in similarities[:10]]
                recommendations = filtered_df.loc[top_indices].copy()
                
                if len(recommendations) > 0:
                    st.subheader(f"🎧 Smart Recommendations ({len(recommendations)} songs)")
                    
                    # Display as expandable cards
                    for i, (idx, song) in enumerate(recommendations.iterrows(), 1):
                        with st.expander(f"{i}. {song['track_name'][:40]}... - {song['artist_name']}", expanded=i<=3):
                            cols = st.columns([3, 2, 1])
                            with cols[0]:
                                st.write(f"**Genre:** {song['genre']}")
                                st.write(f"**Similarity Score:** {similarities[i-1][1]:.3f}")
                                if song['genre'] == target['genre']:
                                    st.success("🎯 Same genre as target!")
                            with cols[1]:
                                st.write(f"**Dance:** {song['danceability']:.2f}")
                                st.write(f"**Energy:** {song['energy']:.2f}")
                                st.write(f"**Happy:** {song['valence']:.2f}")
                            with cols[2]:
                                st.metric("Popularity", f"{song['popularity']}/100")
                                duration = song['duration_ms'] / 60000
                                st.write(f"⏱️ {duration:.1f} min")
                else:
                    st.warning("No similar songs found with current filters. Try relaxing some constraints.")
            else:
                st.error(f"Song '{song_query}' by '{artist_query}' not found!")
                st.info("Try searching for artists in your dataset:")
                # Show some popular artists
                popular_artists = df['artist_name'].value_counts().head(10).index.tolist()
                st.write(", ".join(popular_artists))

with tab2:
    st.header("🎧 Generate Custom Playlist")
    
    playlist_size = st.slider("Playlist Size", 5, 50, 15)
    
    if st.button("✨ Generate Random Playlist", type="primary"):
        # Apply all filters from sidebar
        filtered_df = df.copy()
        
        if selected_genres:
            filtered_df = filtered_df[filtered_df['genre'].isin(selected_genres)]
        
        # Apply mood filter
        mood_filters = mood_options[selected_mood]
        for feature, (min_val, max_val) in mood_filters.items():
            if feature in filtered_df.columns:
                filtered_df = filtered_df[
                    (filtered_df[feature] >= min_val) & 
                    (filtered_df[feature] <= max_val)
                ]
        
        # Apply other filters
        filtered_df = filtered_df[
            (filtered_df['danceability'] >= min_dance) &
            (filtered_df['danceability'] <= max_dance) &
            (filtered_df['energy'] >= min_energy) &
            (filtered_df['energy'] <= max_energy) &
            (filtered_df['valence'] >= min_valence) &
            (filtered_df['valence'] <= max_valence) &
            (filtered_df['tempo'] >= tempo_range[0]) &
            (filtered_df['tempo'] <= tempo_range[1]) &
            (filtered_df['duration_min'] >= duration_range[0]) &
            (filtered_df['duration_min'] <= duration_range[1]) &
            (filtered_df['popularity'] >= 40)  # Only somewhat popular songs
        ]
        
        if len(filtered_df) >= playlist_size:
            # Create a well-balanced playlist
            playlist = filtered_df.sample(n=playlist_size, random_state=42)
            
            # Sort by energy for good flow
            playlist = playlist.sort_values('energy')
            
            st.success(f"✅ Generated {playlist_size}-song playlist")
            
            # Display playlist
            total_duration = playlist['duration_min'].sum()
            st.write(f"**Total Duration:** {total_duration:.1f} minutes")
            st.write(f"**Avg Popularity:** {playlist['popularity'].mean():.1f}/100")
            
            # Create a nice table
            playlist_display = playlist[['track_name', 'artist_name', 'genre', 'popularity', 'danceability', 'energy']].copy()
            playlist_display.columns = ['Song', 'Artist', 'Genre', 'Popularity', 'Dance', 'Energy']
            st.dataframe(playlist_display, use_container_width=True, hide_index=True)
            
            # Download button
            csv = playlist[['track_name', 'artist_name', 'genre', 'danceability', 'energy', 'valence', 'tempo', 'popularity']].to_csv(index=False)
            st.download_button("📥 Download Playlist as CSV", csv, "spotify_playlist.csv", "text/csv")
        else:
            st.warning(f"Only {len(filtered_df)} songs match your criteria. Try relaxing filters.")

with tab3:
    st.header("📊 Dataset Explorer")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Songs", f"{len(df):,}")
    with col2:
        st.metric("Unique Artists", df['artist_name'].nunique())
    with col3:
        st.metric("Genres", df['genre'].nunique())
    
    # Genre distribution
    st.subheader("🎵 Top 10 Genres")
    top_10_genres = df['genre'].value_counts().head(10)
    
    # Create a bar chart with matplotlib
    fig, ax = plt.subplots(figsize=(10, 4))
    top_10_genres.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title("Most Common Genres")
    ax.set_ylabel("Number of Songs")
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Feature distributions
    st.subheader("📈 Audio Feature Distributions")
    feature = st.selectbox("Select feature to visualize", 
                          ['danceability', 'energy', 'valence', 'tempo', 'acousticness', 'loudness'])
    
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.hist(df[feature].dropna(), bins=30, edgecolor='black', alpha=0.7)
    ax2.set_title(f"Distribution of {feature.capitalize()}")
    ax2.set_xlabel(feature.capitalize())
    ax2.set_ylabel("Number of Songs")
    plt.tight_layout()
    st.pyplot(fig2)
    
    # Show feature stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(f"Avg {feature}", f"{df[feature].mean():.2f}")
    with col2:
        st.metric(f"Min {feature}", f"{df[feature].min():.2f}")
    with col3:
        st.metric(f"Max {feature}", f"{df[feature].max():.2f}")
    
    # Sample songs
    st.subheader("🎧 Samples from the dataset")
    
    # Filter options for sample
    sample_filter = st.selectbox("Filter sample by", 
                                ["All", "High Popularity (>80)", "High Danceability (>0.8)", 
                                 "Movie Genre", "Specific Genre"])
    
    sample_df = df.copy()
    if sample_filter == "High Popularity (>80)":
        sample_df = sample_df[sample_df['popularity'] > 80]
    elif sample_filter == "High Danceability (>0.8)":
        sample_df = sample_df[sample_df['danceability'] > 0.8]
    elif sample_filter == "Movie Genre":
        sample_df = sample_df[sample_df['genre'] == 'Movie']
    elif sample_filter == "Specific Genre":
        specific_genre = st.selectbox("Choose genre", sorted(df['genre'].unique()))
        sample_df = sample_df[sample_df['genre'] == specific_genre]
    
    # Display sample
    sample = sample_df.sample(min(10, len(sample_df)))[['track_name', 'artist_name', 'genre', 'popularity', 'danceability', 'energy']]
    sample.columns = ['Song', 'Artist', 'Genre', 'Popularity', 'Dance', 'Energy']
    st.dataframe(sample, use_container_width=True)

# Footer
st.divider()
st.caption("🎯 **Features:** Smart genre-aware recommendations | Mood-based filtering | Custom feature ranges | Movie genre support")
