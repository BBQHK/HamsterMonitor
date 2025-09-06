#!/usr/bin/env python3
"""
Hamster Activity Database Analyzer
A simple script to analyze and visualize data from the hamster_activity.db file
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
import seaborn as sns
from pathlib import Path

class HamsterDatabaseAnalyzer:
    def __init__(self, db_path: str = "hamster_activity.db"):
        """Initialize the analyzer with database path."""
        self.db_path = db_path
        self.conn = None
        self.df = None
        
    def connect(self):
        """Connect to the database and load data."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            print(f"Connected to database: {self.db_path}")
            
            # Load all data into a pandas DataFrame
            query = "SELECT * FROM activity_logs"
            self.df = pd.read_sql_query(query, self.conn)
            
            # Convert timestamp to datetime
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            
            print(f"Loaded {len(self.df)} records")
            print(f"Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
            
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return False
        return True
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            print("Database connection closed")
    
    def basic_stats(self):
        """Display basic statistics about the data."""
        if self.df is None:
            print("No data loaded. Call connect() first.")
            return
        
        print("\n=== BASIC STATISTICS ===")
        print(f"Total records: {len(self.df)}")
        print(f"Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        print(f"Duration: {self.df['timestamp'].max() - self.df['timestamp'].min()}")
        
        print("\nActivity distribution:")
        activity_counts = self.df['activity'].value_counts()
        for activity, count in activity_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {activity}: {count} ({percentage:.1f}%)")
        
        print(f"\nAverage confidence: {self.df['confidence'].mean():.3f}")
        print(f"Average motion intensity: {self.df['motion_intensity'].mean():.3f}")
    
    def plot_activity_timeline(self, figsize=(15, 8)):
        """Plot activity timeline over time."""
        if self.df is None:
            print("No data loaded. Call connect() first.")
            return
        
        plt.figure(figsize=figsize)
        
        # Create a scatter plot of activities over time
        activities = self.df['activity'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(activities)))
        
        for i, activity in enumerate(activities):
            mask = self.df['activity'] == activity
            plt.scatter(self.df[mask]['timestamp'], 
                       [i] * mask.sum(), 
                       c=[colors[i]], 
                       label=activity, 
                       alpha=0.7, 
                       s=50)
        
        plt.yticks(range(len(activities)), activities)
        plt.xlabel('Time')
        plt.ylabel('Activity')
        plt.title('Hamster Activity Timeline')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_confidence_distribution(self, figsize=(12, 8)):
        """Plot confidence distribution by activity."""
        if self.df is None:
            print("No data loaded. Call connect() first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Box plot of confidence by activity
        self.df.boxplot(column='confidence', by='activity', ax=ax1)
        ax1.set_title('Confidence Distribution by Activity')
        ax1.set_xlabel('Activity')
        ax1.set_ylabel('Confidence')
        
        # Histogram of overall confidence
        ax2.hist(self.df['confidence'], bins=30, alpha=0.7, edgecolor='black')
        ax2.set_title('Overall Confidence Distribution')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Frequency')
        ax2.axvline(self.df['confidence'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {self.df["confidence"].mean():.3f}')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_probability_heatmap(self, figsize=(10, 8)):
        """Plot probability heatmap for different activities."""
        if self.df is None:
            print("No data loaded. Call connect() first.")
            return
        
        # Get probability columns
        prob_cols = ['running_prob', 'eating_prob', 'drinking_prob', 'resting_prob', 'exploring_prob']
        
        # Calculate average probabilities by activity
        prob_data = []
        activities = self.df['activity'].unique()
        
        for activity in activities:
            mask = self.df['activity'] == activity
            avg_probs = self.df[mask][prob_cols].mean()
            prob_data.append(avg_probs.values)
        
        # Create heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(prob_data, 
                    xticklabels=[col.replace('_prob', '').title() for col in prob_cols],
                    yticklabels=activities,
                    annot=True, 
                    fmt='.3f',
                    cmap='YlOrRd')
        plt.title('Average Activity Probabilities by Detected Activity')
        plt.xlabel('Activity Type')
        plt.ylabel('Detected Activity')
        plt.tight_layout()
        plt.show()
    
    def plot_motion_analysis(self, figsize=(15, 10)):
        """Plot motion intensity analysis."""
        if self.df is None:
            print("No data loaded. Call connect() first.")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Motion intensity over time
        ax1.plot(self.df['timestamp'], self.df['motion_intensity'], alpha=0.7)
        ax1.set_title('Motion Intensity Over Time')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Motion Intensity')
        ax1.grid(True, alpha=0.3)
        
        # Motion intensity by activity
        self.df.boxplot(column='motion_intensity', by='activity', ax=ax2)
        ax2.set_title('Motion Intensity by Activity')
        ax2.set_xlabel('Activity')
        ax2.set_ylabel('Motion Intensity')
        
        # Motion intensity distribution
        ax3.hist(self.df['motion_intensity'], bins=30, alpha=0.7, edgecolor='black')
        ax3.set_title('Motion Intensity Distribution')
        ax3.set_xlabel('Motion Intensity')
        ax3.set_ylabel('Frequency')
        ax3.axvline(self.df['motion_intensity'].mean(), color='red', linestyle='--',
                    label=f'Mean: {self.df["motion_intensity"].mean():.3f}')
        ax3.legend()
        
        # Motion vs Confidence scatter
        ax4.scatter(self.df['motion_intensity'], self.df['confidence'], alpha=0.6)
        ax4.set_title('Motion Intensity vs Confidence')
        ax4.set_xlabel('Motion Intensity')
        ax4.set_ylabel('Confidence')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_hourly_activity(self, figsize=(12, 8)):
        """Plot activity patterns by hour of day."""
        if self.df is None:
            print("No data loaded. Call connect() first.")
            return
        
        # Extract hour from timestamp
        self.df['hour'] = self.df['timestamp'].dt.hour
        
        plt.figure(figsize=figsize)
        
        # Count activities by hour
        hourly_activity = self.df.groupby(['hour', 'activity']).size().unstack(fill_value=0)
        
        # Plot stacked bar chart
        hourly_activity.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title('Activity Patterns by Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Detections')
        plt.legend(title='Activity', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=0)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_daily_summary(self, figsize=(15, 10)):
        """Plot daily activity summary."""
        if self.df is None:
            print("No data loaded. Call connect() first.")
            return
        
        # Extract date from timestamp
        self.df['date'] = self.df['timestamp'].dt.date
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Daily activity counts
        daily_counts = self.df.groupby('date').size()
        ax1.plot(daily_counts.index, daily_counts.values, marker='o')
        ax1.set_title('Daily Activity Counts')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Number of Detections')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Daily activity distribution
        daily_activity = self.df.groupby(['date', 'activity']).size().unstack(fill_value=0)
        daily_activity.plot(kind='bar', stacked=True, ax=ax2)
        ax2.set_title('Daily Activity Distribution')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Number of Detections')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend(title='Activity', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Daily average confidence
        daily_confidence = self.df.groupby('date')['confidence'].mean()
        ax3.plot(daily_confidence.index, daily_confidence.values, marker='o', color='orange')
        ax3.set_title('Daily Average Confidence')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Average Confidence')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Daily average motion intensity
        daily_motion = self.df.groupby('date')['motion_intensity'].mean()
        ax4.plot(daily_motion.index, daily_motion.values, marker='o', color='green')
        ax4.set_title('Daily Average Motion Intensity')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Average Motion Intensity')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_full_analysis(self):
        """Run all analysis plots."""
        if not self.connect():
            return
        
        print("Starting full analysis...")
        
        # Basic statistics
        self.basic_stats()
        
        # Generate all plots
        self.plot_activity_timeline()
        self.plot_confidence_distribution()
        self.plot_probability_heatmap()
        self.plot_motion_analysis()
        self.plot_hourly_activity()
        self.plot_daily_summary()
        
        self.close()
        print("Analysis complete!")

def main():
    """Main function to run the analyzer."""
    print("Hamster Activity Database Analyzer")
    print("=" * 40)
    
    # Check if database exists
    db_path = "hamster_activity.db"
    if not Path(db_path).exists():
        print(f"Database file '{db_path}' not found!")
        print("Please make sure the database file exists in the current directory.")
        return
    
    # Create analyzer and run analysis
    analyzer = HamsterDatabaseAnalyzer(db_path)
    
    try:
        analyzer.run_full_analysis()
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"Error during analysis: {e}")
    finally:
        analyzer.close()

if __name__ == "__main__":
    main()