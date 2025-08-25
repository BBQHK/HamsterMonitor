import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from activity_logger import ActivityLogger

class ActivityCharts:
    def __init__(self, logger: ActivityLogger):
        """
        Initialize the activity charts generator.
        
        Args:
            logger: ActivityLogger instance
        """
        self.logger = logger
        self.colors = {
            'running': '#FF6B6B',
            'eating': '#4ECDC4',
            'drinking': '#45B7D1',
            'resting': '#96CEB4',
            'exploring': '#FFEAA7'
        }
    
    def create_time_series_chart(self, hours: int = 24, save_path: str = None) -> str:
        """
        Create a time series chart showing activity probabilities over time.
        
        Args:
            hours: Number of hours to analyze
            save_path: Path to save the chart (optional)
            
        Returns:
            Path to saved chart
        """
        data = self.logger.get_activity_data(hours)
        
        if not data:
            print("No data available for the specified time period")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot each activity probability
        activities = ['running', 'eating', 'drinking', 'resting', 'exploring']
        for activity in activities:
            prob_col = f'{activity}_prob'
            if prob_col in df.columns:
                ax.plot(df['timestamp'], df[prob_col], 
                       label=activity.title(), color=self.colors[activity], 
                       linewidth=2, alpha=0.8)
        
        # Add motion intensity
        ax.plot(df['timestamp'], df['motion_intensity'], 
               label='Motion Intensity', color='#9B59B6', 
               linewidth=2, alpha=0.6, linestyle='--')
        
        # Customize the plot
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Probability / Intensity', fontsize=12)
        ax.set_title(f'Hamster Activity Analysis - Last {hours} Hours', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, hours//6)))
        plt.xticks(rotation=45)
        
        # Set y-axis limits
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        # Save the chart
        if save_path is None:
            save_path = f"activity_timeseries_{hours}h_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_activity_distribution_chart(self, hours: int = 24, save_path: str = None) -> str:
        """
        Create a pie chart showing activity distribution.
        
        Args:
            hours: Number of hours to analyze
            save_path: Path to save the chart (optional)
            
        Returns:
            Path to saved chart
        """
        summary = self.logger.get_activity_summary(hours)
        
        if summary['total_records'] == 0:
            print("No data available for the specified time period")
            return None
        
        # Prepare data for pie chart
        activities = list(summary['activity_counts'].keys())
        counts = list(summary['activity_counts'].values())
        colors = [self.colors.get(act, '#CCCCCC') for act in activities]
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart for activity counts
        wedges, texts, autotexts = ax1.pie(counts, labels=activities, colors=colors, 
                                          autopct='%1.1f%%', startangle=90)
        ax1.set_title('Activity Distribution (Count)', fontsize=14, fontweight='bold')
        
        # Bar chart for average probabilities
        avg_probs = summary['avg_probabilities']
        prob_activities = list(avg_probs.keys())
        prob_values = list(avg_probs.values())
        prob_colors = [self.colors.get(act, '#CCCCCC') for act in prob_activities]
        
        bars = ax2.bar(prob_activities, prob_values, color=prob_colors, alpha=0.8)
        ax2.set_title('Average Activity Probabilities', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Average Probability', fontsize=12)
        ax2.set_ylim(0, 1.0)
        
        # Add value labels on bars
        for bar, value in zip(bars, prob_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Save the chart
        if save_path is None:
            save_path = f"activity_distribution_{hours}h_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_heatmap_chart(self, hours: int = 24, save_path: str = None) -> str:
        """
        Create a heatmap showing activity patterns by hour.
        
        Args:
            hours: Number of hours to analyze
            save_path: Path to save the chart (optional)
            
        Returns:
            Path to saved chart
        """
        data = self.logger.get_activity_data(hours)
        
        if not data:
            print("No data available for the specified time period")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        
        # Create heatmap data
        activities = ['running', 'eating', 'drinking', 'resting', 'exploring']
        heatmap_data = []
        
        for hour in range(24):
            hour_data = df[df['hour'] == hour]
            if len(hour_data) > 0:
                row = []
                for activity in activities:
                    prob_col = f'{activity}_prob'
                    avg_prob = hour_data[prob_col].mean()
                    row.append(avg_prob)
                heatmap_data.append(row)
            else:
                heatmap_data.append([0] * len(activities))
        
        heatmap_data = np.array(heatmap_data)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(heatmap_data.T, cmap='YlOrRd', aspect='auto')
        
        # Customize the plot
        ax.set_xticks(range(24))
        ax.set_xticklabels([f'{h:02d}:00' for h in range(24)])
        ax.set_yticks(range(len(activities)))
        ax.set_yticklabels([act.title() for act in activities])
        
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Activity', fontsize=12)
        ax.set_title(f'Activity Heatmap - Last {hours} Hours', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Average Probability', fontsize=12)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save the chart
        if save_path is None:
            save_path = f"activity_heatmap_{hours}h_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_interactive_dashboard(self, hours: int = 24, save_path: str = None) -> str:
        """
        Create an interactive HTML dashboard using Plotly.
        
        Args:
            hours: Number of hours to analyze
            save_path: Path to save the HTML file (optional)
            
        Returns:
            Path to saved HTML file
        """
        data = self.logger.get_activity_data(hours)
        
        if not data:
            print("No data available for the specified time period")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Activity Probabilities Over Time', 'Activity Distribution',
                          'Motion Intensity', 'Activity Summary', 'Hourly Activity Pattern', ''),
            specs=[[{"secondary_y": False}, {"type": "pie"}],
                   [{"secondary_y": False}, {"type": "bar"}],
                   [{"type": "heatmap"}, {"type": "scatter"}]]
        )
        
        # Time series plot
        activities = ['running', 'eating', 'drinking', 'resting', 'exploring']
        for activity in activities:
            prob_col = f'{activity}_prob'
            if prob_col in df.columns:
                fig.add_trace(
                    go.Scatter(x=df['timestamp'], y=df[prob_col],
                              mode='lines', name=activity.title(),
                              line=dict(color=self.colors[activity])),
                    row=1, col=1
                )
        
        # Motion intensity
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['motion_intensity'],
                      mode='lines', name='Motion Intensity',
                      line=dict(color='#9B59B6', dash='dash')),
            row=1, col=1
        )
        
        # Pie chart
        summary = self.logger.get_activity_summary(hours)
        activity_counts = summary['activity_counts']
        fig.add_trace(
            go.Pie(labels=list(activity_counts.keys()),
                   values=list(activity_counts.values()),
                   marker_colors=[self.colors.get(act, '#CCCCCC') for act in activity_counts.keys()]),
            row=1, col=2
        )
        
        # Motion intensity bar chart
        fig.add_trace(
            go.Bar(x=df['timestamp'], y=df['motion_intensity'],
                   name='Motion Intensity', marker_color='#9B59B6'),
            row=2, col=1
        )
        
        # Activity summary bar chart
        avg_probs = summary['avg_probabilities']
        fig.add_trace(
            go.Bar(x=list(avg_probs.keys()), y=list(avg_probs.values()),
                   name='Avg Probabilities',
                   marker_color=[self.colors.get(act, '#CCCCCC') for act in avg_probs.keys()]),
            row=2, col=2
        )
        
        # Heatmap
        df['hour'] = df['timestamp'].dt.hour
        heatmap_data = []
        for hour in range(24):
            hour_data = df[df['hour'] == hour]
            if len(hour_data) > 0:
                row = []
                for activity in activities:
                    prob_col = f'{activity}_prob'
                    avg_prob = hour_data[prob_col].mean()
                    row.append(avg_prob)
                heatmap_data.append(row)
            else:
                heatmap_data.append([0] * len(activities))
        
        fig.add_trace(
            go.Heatmap(z=heatmap_data, x=[f'{h:02d}:00' for h in range(24)],
                      y=[act.title() for act in activities],
                      colorscale='YlOrRd', name='Hourly Pattern'),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'Hamster Activity Dashboard - Last {hours} Hours',
            height=1200,
            showlegend=True
        )
        
        # Save the dashboard
        if save_path is None:
            save_path = f"activity_dashboard_{hours}h_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        fig.write_html(save_path)
        
        return save_path
    
    def generate_all_charts(self, hours: int = 24, output_dir: str = "charts") -> Dict[str, str]:
        """
        Generate all types of charts for the specified time period.
        
        Args:
            hours: Number of hours to analyze
            output_dir: Directory to save charts
            
        Returns:
            Dictionary mapping chart types to file paths
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        charts = {}
        
        try:
            # Time series chart
            charts['time_series'] = self.create_time_series_chart(
                hours, os.path.join(output_dir, f"timeseries_{hours}h.png")
            )
            
            # Distribution chart
            charts['distribution'] = self.create_activity_distribution_chart(
                hours, os.path.join(output_dir, f"distribution_{hours}h.png")
            )
            
            # Heatmap chart
            charts['heatmap'] = self.create_heatmap_chart(
                hours, os.path.join(output_dir, f"heatmap_{hours}h.png")
            )
            
            # Interactive dashboard
            charts['dashboard'] = self.create_interactive_dashboard(
                hours, os.path.join(output_dir, f"dashboard_{hours}h.html")
            )
            
        except Exception as e:
            print(f"Error generating charts: {e}")
        
        return charts 