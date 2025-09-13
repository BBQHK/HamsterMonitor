#!/usr/bin/env python3
"""
Hamster Activity Timeline Analyzer
A simple script to show 24-hour activity timeline from the hamster_activity.db file
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime, date

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
    
    def get_available_dates(self):
        """Get list of available dates in the database."""
        if self.df is None:
            print("No data loaded. Call connect() first.")
            return []
        
        dates = sorted(self.df['timestamp'].dt.date.unique())
        return dates
    
    def filter_by_date(self, target_date):
        """Filter data by specific date."""
        if self.df is None:
            print("No data loaded. Call connect() first.")
            return None
        
        if isinstance(target_date, str):
            target_date = datetime.strptime(target_date, '%Y-%m-%d').date()
        
        filtered_df = self.df[self.df['timestamp'].dt.date == target_date].copy()
        return filtered_df
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            print("Database connection closed")
    
    def basic_stats(self, df=None):
        """Display basic statistics about the data."""
        data_to_use = df if df is not None else self.df
        
        if data_to_use is None:
            print("No data loaded. Call connect() first.")
            return
        
        print("\n=== BASIC STATISTICS ===")
        print(f"Total records: {len(data_to_use)}")
        print(f"Date range: {data_to_use['timestamp'].min()} to {data_to_use['timestamp'].max()}")
        
        print("\nActivity distribution:")
        activity_counts = data_to_use['activity'].value_counts()
        for activity, count in activity_counts.items():
            percentage = (count / len(data_to_use)) * 100
            print(f"  {activity}: {count} ({percentage:.1f}%)")
    
    def plot_24h_activity_timeline(self, figsize=(16, 8), target_date=None):
        """Plot a single timeline chart showing activity detection ranges throughout a 24-hour day."""
        if self.df is None:
            print("No data loaded. Call connect() first.")
            return
        
        # Filter data by date if specified
        if target_date is not None:
            df_to_plot = self.filter_by_date(target_date)
            if df_to_plot is None or len(df_to_plot) == 0:
                print(f"No data found for date: {target_date}")
                return
        else:
            df_to_plot = self.df
        
        # Extract time components
        df_to_plot = df_to_plot.copy()
        df_to_plot['hour'] = df_to_plot['timestamp'].dt.hour
        df_to_plot['minute'] = df_to_plot['timestamp'].dt.minute
        df_to_plot['time_of_day'] = df_to_plot['timestamp'].dt.hour + df_to_plot['timestamp'].dt.minute / 60.0
        
        # Get unique activities and assign colors
        activities = sorted(df_to_plot['activity'].unique())
        if len(activities) == 0:
            print("No activities found in the selected data.")
            return
            
        colors = plt.cm.Set3(np.linspace(0, 1, len(activities)))
        activity_colors = {activity: colors[i] for i, activity in enumerate(activities)}
        
        # Create the plot
        plt.figure(figsize=figsize)
        
        # Merge all activities into one timeline - ABSOLUTELY NO OVERLAPPING
        # Sort all data by timestamp
        all_data = df_to_plot.sort_values('timestamp').copy()
        
        # Create activity ranges on a single line
        y_pos = 0  # Single line at y=0
        ranges = []
        
        # Get all unique timestamps and create non-overlapping segments
        unique_times = sorted(all_data['time_of_day'].unique())
        
        for i, time_point in enumerate(unique_times):
            # Get the activity at this time point
            activity_at_time = all_data[all_data['time_of_day'] == time_point]['activity'].iloc[0]
            
            # Determine start and end times for this segment
            start_time = time_point
            
            if i < len(unique_times) - 1:
                # Not the last time point - end at the next time point
                end_time = unique_times[i + 1]
            else:
                # Last time point - extend slightly for visibility
                end_time = time_point + 0.01  # 36 seconds
        
            ranges.append((start_time, end_time, activity_at_time))
        
        # Plot all ranges on a single line
        for start_time, end_time, activity in ranges:
            # Use exact duration - no minimum width to avoid overlaps
            duration = end_time - start_time
            
            # Only plot if duration is positive
            if duration > 0:
                plt.barh(y_pos, duration, left=start_time, height=0.8, 
                        color=activity_colors[activity], alpha=0.7, 
                        edgecolor='black', linewidth=0.5)
        
        # Customize the plot
        plt.xlabel('Hour of Day (24h)', fontsize=12)
        plt.ylabel('Activity Type', fontsize=12)
        
        # Set title based on whether date is filtered
        if target_date is not None:
            title_date = target_date if isinstance(target_date, str) else target_date.strftime('%Y-%m-%d')
            plt.title(f'Hamster Activity Timeline - {title_date}', fontsize=14, fontweight='bold')
        else:
            plt.title('Hamster Activity Timeline (All Data)', fontsize=14, fontweight='bold')
        
        # Set y-axis for single line
        plt.ylim(-0.5, 0.5)
        plt.yticks([0], ['Activities'])
        
        # Set x-axis to show 24-hour format
        plt.xlim(0, 24)
        plt.xticks(range(0, 25, 2), [f'{h:02d}:00' for h in range(0, 25, 2)])
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add legend showing all activities
        legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=activity_colors[activity], 
                                       alpha=0.7, edgecolor='black', label=activity) 
                          for activity in activities]
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))
        
        # Add time period labels
        plt.axvspan(0, 6, alpha=0.1, color='blue', label='Night')
        plt.axvspan(6, 12, alpha=0.1, color='yellow', label='Morning')
        plt.axvspan(12, 18, alpha=0.1, color='orange', label='Afternoon')
        plt.axvspan(18, 24, alpha=0.1, color='purple', label='Evening')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\n=== 24-HOUR ACTIVITY SUMMARY ===")
        for activity in activities:
            activity_data = df_to_plot[df_to_plot['activity'] == activity]
            total_detections = len(activity_data)
            if total_detections > 0:
                avg_time = activity_data['time_of_day'].mean()
                std_time = activity_data['time_of_day'].std()
                print(f"{activity}: {total_detections} detections, avg time: {avg_time:.1f}h ± {std_time:.1f}h")
    
    def run_analysis(self, target_date=None):
        """Run the 24-hour timeline analysis."""
        if not self.connect():
            return
        
        print("Starting 24-hour timeline analysis...")
        
        # Show available dates
        available_dates = self.get_available_dates()
        print(f"\nAvailable dates: {[d.strftime('%Y-%m-%d') for d in available_dates]}")
        
        # Basic statistics
        if target_date is not None:
            filtered_df = self.filter_by_date(target_date)
            if filtered_df is not None and len(filtered_df) > 0:
                print(f"\nData for {target_date}:")
                self.basic_stats(filtered_df)
            else:
                print(f"No data found for date: {target_date}")
                return
        else:
            self.basic_stats()
        
        # Generate the timeline plot
        self.plot_24h_activity_timeline(target_date=target_date)
        
        self.close()
        print("Analysis complete!")

def main():
    """Main function to run the analyzer."""
    print("Hamster Activity Timeline Analyzer")
    print("=" * 40)
    
    # Check if database exists
    db_path = "hamster_activity.db"
    if not Path(db_path).exists():
        print(f"Database file '{db_path}' not found!")
        print("Please make sure the database file exists in the current directory.")
        return
    
    # Create analyzer and connect to get available dates
    analyzer = HamsterDatabaseAnalyzer(db_path)
    
    try:
        # Connect to get available dates
        if not analyzer.connect():
            return
        
        available_dates = analyzer.get_available_dates()
        analyzer.close()
        
        if not available_dates:
            print("No data found in the database.")
            return
        
        # Ask user for date selection
        print(f"\nAvailable dates: {[d.strftime('%Y-%m-%d') for d in available_dates]}")
        print("\nOptions:")
        print("1. Press Enter to show all data combined")
        print("2. Enter a specific date (YYYY-MM-DD format)")
        print("3. Type 'latest' to show the most recent date")
        
        user_input = input("\nEnter your choice: ").strip().lower()
        
        target_date = None
        if user_input == 'latest':
            target_date = available_dates[-1]
            print(f"Selected latest date: {target_date}")
        elif user_input and user_input != '':
            try:
                # Try to parse the date
                parsed_date = datetime.strptime(user_input, '%Y-%m-%d').date()
                if parsed_date in available_dates:
                    target_date = parsed_date
                    print(f"Selected date: {target_date}")
                else:
                    print(f"Date {user_input} not found in available dates.")
                    print("Showing all data instead.")
            except ValueError:
                print(f"Invalid date format: {user_input}")
                print("Please use YYYY-MM-DD format. Showing all data instead.")
        
        # Run analysis with selected date
        analyzer.run_analysis(target_date)
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"Error during analysis: {e}")
    finally:
        analyzer.close()

if __name__ == "__main__":
    main()