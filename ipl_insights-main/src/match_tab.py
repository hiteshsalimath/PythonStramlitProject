import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 


def total_match(matches, years, team) -> int:
    return len(matches[(matches['year'].isin(years)) & (matches['teams'].str.contains(team))])


def total_win(matches, years: object, team: object) -> int:
    return len(matches[
                   (matches['year'].isin(years)) & (matches['teams'].str.contains(team)) & (matches['winner'] == team)])


def total_lost(matches, years, team) -> int:
    return len(matches[
                   (matches['year'].isin(years)) & (matches['teams'].str.contains(team)) & (
                           matches['winner'] != team) & (
                       matches['winner'].notna())])


def display_match_tab(tab, matches, selected_team, selected_years):
    with tab:
        col = st.columns((10, 8), gap='small')
        match_count = total_match(matches, selected_years, selected_team)
        win_count = total_win(matches, selected_years, selected_team)
        lost_count = total_lost(matches, selected_years, selected_team)
        tie_count = match_count - win_count - lost_count

        # Pie Chart
        with col[0]:
            st.markdown(f"# Overall Performance")
            fig = go.Figure(
                go.Pie(
                    hole=0.5,
                    textinfo='label+value',
                    hoverinfo='label+percent',
                    labels=['Won', 'Lost', 'Tied'],
                    values=[win_count, lost_count, tie_count],
                    marker=dict(colors=['#2ca02c', '#d62728', '#ff7f0e'])
                ))
            fig.update_layout(width=300, height=300, margin=dict(l=0, r=0, t=30, b=10, pad=0),
                              annotations=[
                                  dict(text=f'Matches: {match_count}', x=0.5, y=0.5, font_size=15, showarrow=False)])
            st.plotly_chart(fig)

        
            
        # 3D Stacked Bar Plot
        with col[0]:
            st.markdown("\n"f"## Performance over the seasons")
            win_data = [total_win(matches, [year], selected_team) for year in selected_years]
            lost_data = [total_lost(matches, [year], selected_team) for year in selected_years]
            tie_data = [total_match(matches, [year], selected_team) - total_win(matches, [year], selected_team) - total_lost(
                        matches, [year], selected_team)
                        for year in selected_years]

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')

            # Position of bars on x-axis
            x_pos = np.arange(len(selected_years))

            # Plotting stacked bars
            ax.bar(x_pos, win_data, zs=0, zdir='y', color='b', alpha=0.8, label='Won')
            ax.bar(x_pos, lost_data, zs=1, zdir='y', color='r', alpha=0.8, label='Lost')
            ax.bar(x_pos, tie_data, zs=2, zdir='y', color='g', alpha=0.8, label='Tied')

            # Labels and title
            ax.set_xlabel('Seasons')
            ax.set_ylabel('Match Outcome')
            ax.set_zlabel('Number of Matches',rotation = 90)
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(['Won', 'Lost', 'Tied'])
            ax.set_xticks(x_pos)
            ax.set_xticklabels(selected_years)
            ax.set_title('Overall Performance')

            # Adding legend
            ax.legend()

            st.pyplot(fig)



        st.markdown("\n"f"## Impact of Toss Results on Match Victories")
        col = st.columns((1, 8), gap='small')
        with col[0]:
            toss_result = st.radio(
                ":blue[Toss Result]",
                ["Won", "Lost"])

            toss_decision = st.radio(
                ":blue[Toss Decision]",
                ["Bat", "Field", "Any"])

            # Filter matches based on selected years and team
            filtered_matches = matches[
                (matches['year'].isin(selected_years)) &
                (matches['teams'].str.contains(selected_team))
                ]

            # Filter matches based on toss result
            if toss_result == "Lost":
                filtered_matches = filtered_matches[filtered_matches['toss_winner'] != selected_team]
            else:
                filtered_matches = filtered_matches[filtered_matches['toss_winner'] == selected_team]

            match_count_on_toss_result = len(filtered_matches)

            # Filter matches based on toss decision
            if toss_decision == "Field":
                filtered_matches = filtered_matches[filtered_matches['toss_decision'] == 'field']
            elif toss_decision == "Bat":
                filtered_matches = filtered_matches[filtered_matches['toss_decision'] == 'bat']

            win_match_count = len(filtered_matches[filtered_matches['winner'] == selected_team])
        
        
        
        
        with col[1]:
            fig = go.Figure(
                go.Pie(
                    hole=0.8,
                    textinfo='value',
                    hoverinfo='label+value',
                    labels=['Won', 'Lost/Tie'],
                    values=[win_match_count, match_count_on_toss_result - win_match_count],
                    showlegend=False,
                    marker=dict(colors=['#2ca02c', '#fff'], line=dict(color='#000000', width=0.2))
                ))
            winning_percent = (win_match_count / match_count_on_toss_result) * 100
            fig.update_layout(width=220, height=220, margin=dict(l=0, r=0, t=0, b=10, pad=0),
                              annotations=[
                                  dict(text="{:.0f}%".format(winning_percent), x=0.5, y=0.5, font_size=50, showarrow=False)])
            st.plotly_chart(fig)
