import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import percentileofscore
from nba_api.stats.endpoints import leaguedashplayerstats
from nba_api.stats.static import players
import time
import warnings
warnings.filterwarnings('ignore')
import requests
from PIL import Image
from io import BytesIO


class NBAPlayerComparison:
    def __init__(self):
        self.player_data = None
        
    def fetch_player_data(self, season='2023-24', season_type='Regular Season'):
        """Fetch player statistics from NBA API"""
        try:
            # Get league player stats
            player_stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                season_type_all_star=season_type
            )
            
            df = player_stats.get_data_frames()[0]
            
            # Select and process relevant columns
            stats_columns = [
                'PLAYER_NAME', 'TEAM_ABBREVIATION', 'GP', 'MIN', 'FGM', 'FGA', 'FG_PCT',
                'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB',
                'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'PLUS_MINUS'
            ]
            
            df = df[stats_columns]
            
            # Filter players with minimum games played
            min_games = 20 if season_type == 'Regular Season' else 5
            df = df[df['GP'] >= min_games]
            
            # Create per-game and advanced statistics
            df['PPG'] = df['PTS'] / df['GP']
            df['RPG'] = df['REB'] / df['GP']
            df['APG'] = df['AST'] / df['GP']
            df['SPG'] = df['STL'] / df['GP']
            df['BPG'] = df['BLK'] / df['GP']
            df['TPG'] = df['TOV'] / df['GP']
            df['MPG'] = df['MIN'] / df['GP']
            
            # Advanced metrics
            df['TS_PCT'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA']))
            df['AST_TO_RATIO'] = np.where(df['TOV'] > 0, df['AST'] / df['TOV'], df['AST'])
            df['USAGE_RATE'] = (df['FGA'] + 0.44 * df['FTA'] + df['TOV']) / df['GP']
            
            # Replace infinite values and handle NaN
            df = df.replace([np.inf, -np.inf], 0)
            df = df.fillna(0)
            
            # Ensure shooting percentages are reasonable
            df['FG_PCT'] = df['FG_PCT'].clip(0, 1)
            df['FG3_PCT'] = df['FG3_PCT'].clip(0, 1)
            df['FT_PCT'] = df['FT_PCT'].clip(0, 1)
            df['TS_PCT'] = df['TS_PCT'].clip(0, 1)
            
            self.player_data = df
            return df
            
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None
        

    def get_player_id(self, player_name):
        """Get NBA player ID from player name"""
        try:
            # Find player in NBA API
            player_list = players.find_players_by_full_name(player_name)
            if player_list:
                return player_list[0]['id']
            return None
        except:
            return None
        
    def load_player_image(self, player_id):
        """Load player headshot image from NBA CDN"""
        try:
            image_url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"
            response = requests.get(image_url, timeout=5)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content))
            return None
        except:
            return None
    

    
    def calculate_percentiles(self, df, stat_columns):
        """Calculate true percentiles for each stat"""
        percentile_df = df.copy()
        
        for col in stat_columns:
            # Calculate percentile for each player in this stat
            percentile_df[f'{col}_percentile'] = df[col].apply(
                lambda x: percentileofscore(df[col], x, kind='rank') / 100
            )
        
        return percentile_df
    
    def create_spider_chart(self, selected_players):
        """Create an improved spider web chart with true percentiles"""
        if len(selected_players) < 1:
            return None
            
        # Define spider chart categories with display names and descriptions
        spider_categories = {
            'Scoring': ('PPG', 'Points Per Game'),
            'Rebounding': ('RPG', 'Rebounds Per Game'), 
            'Playmaking': ('APG', 'Assists Per Game'),
            'Steals': ('SPG', 'Steals Per Game'),
            'Shot Blocking': ('BPG', 'Blocks Per Game'),
            'Field Goal %': ('FG_PCT', 'Field Goal Percentage'),
            '3-Point %': ('FG3_PCT', '3-Point Percentage'),
            'Free Throw %': ('FT_PCT', 'Free Throw Percentage'),
            'True Shooting %': ('TS_PCT', 'True Shooting Percentage'),
            'Assist/TO Ratio': ('AST_TO_RATIO', 'Assist to Turnover Ratio')
        }
        
        # Filter data for selected players
        comparison_df = self.player_data[self.player_data['PLAYER_NAME'].isin(selected_players)].copy()
        
        if comparison_df.empty:
            return None
            
        # Get the stat columns
        stat_columns = [info[0] for info in spider_categories.values()]
        
        # Calculate true percentiles
        percentile_df = self.calculate_percentiles(self.player_data, stat_columns)
        comparison_percentiles = percentile_df[percentile_df['PLAYER_NAME'].isin(selected_players)].copy()
        
        # Create the spider chart
        fig = go.Figure()
        
        # NBA-inspired color palette
        nba_colors = [
            '#1D428A',  # NBA Blue
            '#C8102E',  # NBA Red
            '#007A33',  # Boston Celtic Green
            '#FFC72C',  # Lakers Gold
            '#CE1141',  # Miami Heat Red
            '#00538C',  # Dallas Mavericks Blue
            '#007AC1',  # Orlando Magic Blue
            '#00471B',  # Milwaukee Bucks Green
            '#EE2944',  # Chicago Bulls Red
            '#724C9F'   # Sacramento Kings Purple
        ]
        
        # Create spider traces for each player
        for i, player in enumerate(selected_players):
            player_data = comparison_percentiles[comparison_percentiles['PLAYER_NAME'] == player]
            
            if not player_data.empty:
                # Get percentile values
                values = []
                hover_text = []
                raw_values = comparison_df[comparison_df['PLAYER_NAME'] == player].iloc[0]
                
                for category, (stat, description) in spider_categories.items():
                    percentile_val = player_data[f'{stat}_percentile'].values[0]
                    raw_val = raw_values[stat]
                    
                    values.append(percentile_val)
                    
                    # Format hover text based on stat type
                    if stat in ['FG_PCT', 'FG3_PCT', 'FT_PCT', 'TS_PCT']:
                        hover_text.append(f"{description}: {raw_val:.1%}")
                    elif stat == 'AST_TO_RATIO':
                        hover_text.append(f"{description}: {raw_val:.1f}:1")
                    else:
                        hover_text.append(f"{description}: {raw_val:.1f}")
                
                # Close the spider web
                values.append(values[0])
                hover_text.append(hover_text[0])
                categories = list(spider_categories.keys()) + [list(spider_categories.keys())[0]]
                
                # Add trace with NBA styling
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=player,
                    line=dict(
                        color=nba_colors[i % len(nba_colors)],
                        width=4
                    ),
                    fillcolor=nba_colors[i % len(nba_colors)],
                    opacity=0.25,
                    marker=dict(
                        size=10,
                        color=nba_colors[i % len(nba_colors)],
                        line=dict(color='white', width=3)
                    ),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                 '<span style="color:' + nba_colors[i % len(nba_colors)] + '">●</span> %{text}<br>' +
                                 'Percentile: %{r:.0%}<br>' +
                                 '<extra></extra>',
                    text=hover_text
                ))
        
        # Enhanced NBA-style layout
        fig.update_layout(
            polar=dict(
                bgcolor='rgba(248, 248, 248, 0.9)',
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(size=12, color='#1D428A', family='Arial Bold'),
                    gridcolor='rgba(29, 66, 138, 0.2)',
                    linecolor='rgba(29, 66, 138, 0.3)',
                    tickmode='linear',
                    tick0=0,
                    dtick=0.25,
                    showticklabels=True,
                    tickformat='.0%',
                    tickvals=[0, 0.25, 0.5, 0.75, 1],
                    ticktext=['0%', '25%', '50%', '75%', '100%']
                ),
                angularaxis=dict(
                    tickfont=dict(size=14, color='#1D428A', family='Arial Bold'),
                    linecolor='rgba(29, 66, 138, 0.4)',
                    gridcolor='rgba(29, 66, 138, 0.2)',
                    rotation=90,
                    direction='clockwise'
                )
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
                font=dict(size=16, color='#1D428A', family='Arial Bold'),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='rgba(29, 66, 138, 0.5)',
                borderwidth=2
            ),
            title=dict(
                text="<b>NBA Player Performance Comparison</b><br><sup>True Percentile Rankings Across Key Categories</sup>",
                x=0.5,
                font=dict(size=24, color='#1D428A', family='Arial Bold'),
                xanchor='center'
            ),
            paper_bgcolor='rgba(255,255,255,1)',
            plot_bgcolor='rgba(255,255,255,1)',
            font=dict(family='Arial', size=14, color='#1D428A'),
            margin=dict(l=100, r=100, t=120, b=100),
            width=800,
            height=700
        )
        
        return fig, comparison_df
    
    def create_stats_table(self, comparison_df):
        """Create a comprehensive stats comparison table"""
        if comparison_df.empty:
            return None
            
        # Select key stats for display
        display_stats = [
            'PLAYER_NAME', 'TEAM_ABBREVIATION', 'GP', 'MPG', 'PPG', 'RPG', 'APG', 
            'SPG', 'BPG', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'TS_PCT', 'AST_TO_RATIO'
        ]
        
        table_df = comparison_df[display_stats].copy()
        
        # Round numerical values for better display
        numerical_cols = ['MPG', 'PPG', 'RPG', 'APG', 'SPG', 'BPG', 'AST_TO_RATIO']
        percentage_cols = ['FG_PCT', 'FG3_PCT', 'FT_PCT', 'TS_PCT']
        
        for col in numerical_cols:
            table_df[col] = table_df[col].round(1)
        
        for col in percentage_cols:
            table_df[col] = table_df[col].round(3)
        
        # Rename columns for better display
        table_df.columns = [
            'Player', 'Team', 'Games', 'Min/Game', 'Pts/Game', 'Reb/Game', 'Ast/Game',
            'Stl/Game', 'Blk/Game', 'FG%', '3P%', 'FT%', 'TS%', 'AST/TO'
        ]
        
        return table_df
    
    def style_dataframe(self, df):
        """Apply styling to highlight highest and lowest values"""
        def highlight_extremes(s):
            """Highlight max and min values in each column"""
            if s.name in ['PLAYER_NAME', 'TEAM_ABBREVIATION']:
                return [''] * len(s)
            
            # Convert to numeric if possible
            try:
                numeric_s = pd.to_numeric(s, errors='coerce')
                if numeric_s.isna().all():
                    return [''] * len(s)
                
                styles = [''] * len(s)
                max_idx = numeric_s.idxmax()
                min_idx = numeric_s.idxmin()
                
                # Apply styles
                styles[s.index.get_loc(max_idx)] = 'background-color: #90EE90; font-weight: bold'  # Light green for max
                styles[s.index.get_loc(min_idx)] = 'background-color: #FFB6C1; font-weight: bold'  # Light pink for min
                
                return styles
            except:
                return [''] * len(s)
        
        return df.style.apply(highlight_extremes, axis=0)

@st.cache_data
def load_player_data(season, season_type):
    """Cache player data to avoid repeated API calls"""
    comparison = NBAPlayerComparison()
    return comparison.fetch_player_data(season, season_type), comparison

def main():
    st.set_page_config(
        page_title="NBA Player Comparison",
        page_icon="🏀",
        layout="wide",
        initial_sidebar_state="expanded"

    )
    
    # Custom CSS for NBA styling
    st.markdown("""
        <style>
        .main-title {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            color: #1D428A;
            margin-bottom: 1rem;
        }
        
        .subtitle {
            font-size: 1.2rem;
            text-align: center;
            color: #C8102E;
            margin-bottom: 2rem;
        }
        
        .nba-metric {
            background: linear-gradient(135deg, #1D428A 0%, #C8102E 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            font-weight: bold;
        }
        
        .stSelectbox > div > div {
            background-color: #f8f9fa;
            border: 2px solid #1D428A;
            border-radius: 10px;
        }
        
        .info-box {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #1D428A;
            margin: 1rem 0;
        }
        
        .percentile-explanation {
            background-color: #e8f4f8;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #1D428A;
            margin: 1rem 0;
        }
        
        .legend-box {
            background-color: #fff3cd;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #ffc107;
            margin: 1rem 0;
        }
                
        </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.markdown('<h1 class="main-title">NBA Comparison Tool</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Compare your favorite players instantly.</p>', unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.image("poster_comparison_tool.png", width=600)

    st.sidebar.header("Settings")
    
    # Season selection
    season = st.sidebar.selectbox(
        "📅 Select Season",
        ['2024-25','2023-24', '2022-23', '2021-22', '2020-21', '2019-20'],
        index=0
    )
    
    # Season type selection
    season_type = st.sidebar.selectbox(
        "🏆 Season Type",
        ['Regular Season', 'Playoffs'],
        index=0
    )
    
    # Auto-load data
    with st.spinner("🔄 Loading NBA player data..."):
        df, comparison = load_player_data(season, season_type)
    
    if df is not None and not df.empty:
        st.toast(f"✅ Loaded {len(df)} players from {season} {season_type}")
        
        # Explanation of percentile system
        

        
        
        # Player selection
        #st.subheader("🎯 Select Players to Compare")
        
        # Create columns for better layout
        col1, col2 = st.columns([4, 1])
        
        with col1:
            available_players = sorted(df['PLAYER_NAME'].unique())
            selected_players = st.multiselect(
                "", 
                available_players,
                default=[],
                placeholder="Search NBA players—pick up to five",  # text inside the box

                help="Select players to see their performance comparison."
            )


        
        with col2:
            players_count = len(selected_players)
            
            if players_count > 5:
                st.warning("⚠️ Maximum 5 players")
                selected_players = selected_players[:5]
        
        # Display comparison if players are selected
        if selected_players:
            #st.subheader("📊 Performance Comparison Spider Chart")
            

            
            # Create the spider chart
            fig, comparison_df = comparison.create_spider_chart(selected_players)
            
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
                

                
                # Performance insights
                st.subheader("Key Performance Metrics")

                # Create metrics columns
                metrics_cols = st.columns(len(selected_players))

                for i, player in enumerate(selected_players):
                    player_stats = comparison_df[comparison_df['PLAYER_NAME'] == player].iloc[0]
                    
                    with metrics_cols[i]:
                        # Get player ID and load image
                        player_id = comparison.get_player_id(player)
                        if player_id:
                            player_image = comparison.load_player_image(player_id)
                            if player_image:
                                # Resize image to reasonable size
                                #player_image = player_image.resize((150, 150))
                                st.image(player_image, use_container_width=True)
                            else:
                                st.write("📷 Image not available")
                        
                        st.markdown(f"**{player}** ({player_stats['TEAM_ABBREVIATION']})")
                        
                        # Key metrics (rest remains the same)
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("PPG", f"{player_stats['PPG']:.1f}")
                            st.metric("RPG", f"{player_stats['RPG']:.1f}")
                            st.metric("APG", f"{player_stats['APG']:.1f}")
                        
                        with col_b:
                            st.metric("FG%", f"{player_stats['FG_PCT']:.1%}")
                            st.metric("3P%", f"{player_stats['FG3_PCT']:.1%}")
                            st.metric("TS%", f"{player_stats['TS_PCT']:.1%}")



                                # Statistics table
                st.subheader("Detailed Statistics")
                
                stats_table = comparison.create_stats_table(comparison_df)

                if stats_table is not None:
                    # Format the dataframe to reduce decimal places
                    formatted_stats = stats_table.copy()
                    
                    # Define columns and their formatting
                    percentage_cols = ['FG%', '3P%', 'FT%', 'TS%']
                    ratio_cols = ['AST/TO']
                    numeric_cols = ['Min/Game', 'Pts/Game', 'Reb/Game', 'Ast/Game', 'Stl/Game', 'Blk/Game']
                    
                    # Apply formatting
                    for col in percentage_cols:
                        if col in formatted_stats.columns:
                            formatted_stats[col] = formatted_stats[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "")
                    
                    for col in ratio_cols:
                        if col in formatted_stats.columns:
                            formatted_stats[col] = formatted_stats[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "")
                    
                    for col in numeric_cols:
                        if col in formatted_stats.columns:
                            formatted_stats[col] = formatted_stats[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "")
                    
                    # Inline highlighting function
                    def highlight_extremes(s):
                        if s.name in ['Player', 'Team']:
                            return [''] * len(s)
                        try:
                            # Convert back to numeric for comparison (remove formatting)
                            if '%' in str(s.iloc[0]):
                                numeric_s = pd.to_numeric(s.str.replace('%', ''), errors='coerce') / 100
                            else:
                                numeric_s = pd.to_numeric(s, errors='coerce')
                                
                            if numeric_s.isna().all() or numeric_s.nunique() <= 1:
                                return [''] * len(s)
                            
                            max_val, min_val = numeric_s.max(), numeric_s.min()
                            styles = []
                            for val in numeric_s:
                                if pd.isna(val):
                                    styles.append('')
                                elif val == max_val:
                                    styles.append('background-color: #90EE90; font-weight: bold; color: #006400')
                                elif val == min_val:
                                    styles.append('background-color: #FFB6C1; font-weight: bold; color: #8B0000')
                                else:
                                    styles.append('')
                            return styles
                        except:
                            return [''] * len(s)
                    
                    # Apply styling and display
                    styled_table = formatted_stats.style.apply(highlight_extremes, axis=0)
                    
                    st.dataframe(
                        styled_table,
                        use_container_width=True,
                        hide_index=True
                    )


                
            else:
                st.error("❌ Error creating comparison chart")
                
        else:
            #st.info("👆 Select players above to start comparing their performance")
            
            # Show top performers with highlighting
            st.subheader("Top Performers This Season")
            

            
            # Create tabs for different categories
            tab1, tab2, tab3, tab4 = st.tabs(["🏀 Scoring", "📈 Rebounding", "🎯 Assists", "🛡️ Defense"])
            
            with tab1:
                top_scorers = df.nlargest(10, 'PPG')[['PLAYER_NAME', 'TEAM_ABBREVIATION', 'PPG', 'FG_PCT', 'FG3_PCT']].round(3)
                styled_scorers = comparison.style_dataframe(top_scorers)
                st.dataframe(styled_scorers, hide_index=True, use_container_width=True)
            
            with tab2:
                top_rebounders = df.nlargest(10, 'RPG')[['PLAYER_NAME', 'TEAM_ABBREVIATION', 'RPG', 'PPG', 'APG']].round(1)
                styled_rebounders = comparison.style_dataframe(top_rebounders)
                st.dataframe(styled_rebounders, hide_index=True, use_container_width=True)
            
            with tab3:
                top_assists = df.nlargest(10, 'APG')[['PLAYER_NAME', 'TEAM_ABBREVIATION', 'APG', 'AST_TO_RATIO', 'PPG']].round(1)
                styled_assists = comparison.style_dataframe(top_assists)
                st.dataframe(styled_assists, hide_index=True, use_container_width=True)
            
            with tab4:
                # Create a defensive score
                df['DEF_SCORE'] = df['SPG'] + df['BPG']
                top_defense = df.nlargest(10, 'DEF_SCORE')[['PLAYER_NAME', 'TEAM_ABBREVIATION', 'SPG', 'BPG', 'DEF_SCORE']].round(1)
                styled_defense = comparison.style_dataframe(top_defense)
                st.dataframe(styled_defense, hide_index=True, use_container_width=True)
    
    else:
        st.error("❌ Failed to load player data. Please check your connection and try again.")

if __name__ == "__main__":
    main()