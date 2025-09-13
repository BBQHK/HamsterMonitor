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
    
    def detect_abnormal_activities(self, df=None, sensitivity=2.0):
        """
        Detect abnormal activities based on statistical analysis.
        
        Parameters:
        - df: DataFrame to analyze (uses self.df if None)
        - sensitivity: Standard deviation multiplier for anomaly detection (default 2.0)
        
        Returns:
        - Dictionary with abnormal activity information
        """
        data_to_use = df if df is not None else self.df
        
        if data_to_use is None:
            print("No data loaded. Call connect() first.")
            return None
        
        print(f"\n=== ABNORMAL ACTIVITY DETECTION (sensitivity: {sensitivity}σ) ===")
        
        # Add time components for analysis
        analysis_df = data_to_use.copy()
        analysis_df['hour'] = analysis_df['timestamp'].dt.hour
        analysis_df['minute'] = analysis_df['timestamp'].dt.minute
        analysis_df['time_of_day'] = analysis_df['timestamp'].dt.hour + analysis_df['timestamp'].dt.minute / 60.0
        
        abnormal_activities = {
            'unusual_timing': [],
            'excessive_duration': [],
            'missing_activities': [],
            'frequency_anomalies': []
        }
        
        # 1. Detect unusual timing patterns
        print("\n1. UNUSUAL TIMING PATTERNS:")
        for activity in analysis_df['activity'].unique():
            activity_data = analysis_df[analysis_df['activity'] == activity]
            
            if len(activity_data) < 3:  # Need at least 3 data points for statistical analysis
                continue
                
            # Calculate normal time range (mean ± sensitivity * std)
            mean_time = activity_data['time_of_day'].mean()
            std_time = activity_data['time_of_day'].std()
            
            if std_time > 0:  # Avoid division by zero
                lower_bound = mean_time - sensitivity * std_time
                upper_bound = mean_time + sensitivity * std_time
                
                # Find activities outside normal time range
                unusual_times = activity_data[
                    (activity_data['time_of_day'] < lower_bound) | 
                    (activity_data['time_of_day'] > upper_bound)
                ]
                
                if len(unusual_times) > 0:
                    print(f"  {activity}: {len(unusual_times)} unusual timings")
                    for _, row in unusual_times.iterrows():
                        time_str = f"{int(row['time_of_day']):02d}:{int((row['time_of_day'] % 1) * 60):02d}"
                        print(f"    - {time_str} (normal range: {int(lower_bound):02d}:{int((lower_bound % 1) * 60):02d} - {int(upper_bound):02d}:{int((upper_bound % 1) * 60):02d})")
                    
                    abnormal_activities['unusual_timing'].extend([
                        {
                            'activity': activity,
                            'timestamp': row['timestamp'],
                            'time_of_day': row['time_of_day'],
                            'normal_range': (lower_bound, upper_bound),
                            'deviation': abs(row['time_of_day'] - mean_time) / std_time if std_time > 0 else 0
                        }
                        for _, row in unusual_times.iterrows()
                    ])
        
        # 2. Detect excessive activity duration (if we have consecutive data)
        print("\n2. ACTIVITY DURATION ANALYSIS:")
        # Group consecutive activities and calculate durations
        analysis_df = analysis_df.sort_values('timestamp')
        activity_groups = []
        current_group = []
        
        for _, row in analysis_df.iterrows():
            if not current_group or current_group[-1]['activity'] == row['activity']:
                current_group.append(row)
            else:
                if len(current_group) > 0:
                    activity_groups.append(current_group)
                current_group = [row]
        
        if len(current_group) > 0:
            activity_groups.append(current_group)
        
        # Calculate durations and detect anomalies
        durations_by_activity = {}
        for group in activity_groups:
            if len(group) < 2:
                continue
                
            activity = group[0]['activity']
            start_time = group[0]['time_of_day']
            end_time = group[-1]['time_of_day']
            duration = end_time - start_time
            
            if activity not in durations_by_activity:
                durations_by_activity[activity] = []
            durations_by_activity[activity].append(duration)
        
        for activity, durations in durations_by_activity.items():
            if len(durations) < 3:
                continue
                
            mean_duration = np.mean(durations)
            std_duration = np.std(durations)
            
            if std_duration > 0:
                threshold = mean_duration + sensitivity * std_duration
                excessive_durations = [d for d in durations if d > threshold]
                
                if excessive_durations:
                    print(f"  {activity}: {len(excessive_durations)} excessive durations")
                    print(f"    - Normal: {mean_duration:.2f}h ± {std_duration:.2f}h")
                    print(f"    - Excessive: {[f'{d:.2f}h' for d in excessive_durations]}")
                    
                    abnormal_activities['excessive_duration'].extend([
                        {
                            'activity': activity,
                            'duration': duration,
                            'normal_mean': mean_duration,
                            'normal_std': std_duration,
                            'deviation': (duration - mean_duration) / std_duration
                        }
                        for duration in excessive_durations
                    ])
        
        # 3. Detect missing activities (if we have historical data)
        print("\n3. MISSING ACTIVITY ANALYSIS:")
        if len(analysis_df) > 0:
            # Check if certain activities are completely missing
            all_activities = set(analysis_df['activity'].unique())
            
            # Define expected activities (you can modify this list)
            expected_activities = {'eating', 'drinking', 'exploring', 'resting', 'wheel'}
            missing_activities = expected_activities - all_activities
            
            if missing_activities:
                print(f"  Missing activities: {', '.join(missing_activities)}")
                abnormal_activities['missing_activities'] = list(missing_activities)
        
        # 4. Frequency anomalies
        print("\n4. FREQUENCY ANOMALIES:")
        activity_counts = analysis_df['activity'].value_counts()
        total_activities = len(analysis_df)
        
        for activity, count in activity_counts.items():
            frequency = count / total_activities
            
            # Define normal frequency ranges (you can adjust these)
            normal_frequencies = {
                'eating': (0.05, 0.25),    # 5-25% of time
                'drinking': (0.02, 0.10),   # 2-10% of time
                'exploring': (0.10, 0.40),  # 10-40% of time
                'resting': (0.30, 0.70),    # 30-70% of time
                'wheel': (0.05, 0.30)       # 5-30% of time
            }
            
            if activity in normal_frequencies:
                min_freq, max_freq = normal_frequencies[activity]
                if frequency < min_freq or frequency > max_freq:
                    print(f"  {activity}: {frequency:.1%} frequency (normal: {min_freq:.1%}-{max_freq:.1%})")
                    abnormal_activities['frequency_anomalies'].append({
                        'activity': activity,
                        'frequency': frequency,
                        'normal_range': (min_freq, max_freq),
                        'type': 'too_low' if frequency < min_freq else 'too_high'
                    })
        
        return abnormal_activities
    
    def plot_24h_activity_timeline(self, figsize=(16, 8), target_date=None, show_abnormal=True):
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
        
        # Get abnormal activities for highlighting
        abnormal_activities = None
        if show_abnormal:
            abnormal_activities = self.detect_abnormal_activities(df_to_plot, sensitivity=2.0)
        
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
        
        # Merge all activities into one timeline - GROUP CONSECUTIVE SAME ACTIVITIES
        # Sort all data by timestamp
        all_data = df_to_plot.sort_values('timestamp').copy()
        
        # Create activity ranges on a single line
        y_pos = 0  # Single line at y=0
        ranges = []
        
        # Get all unique timestamps and create non-overlapping segments
        unique_times = sorted(all_data['time_of_day'].unique())
        
        # Group consecutive activities of the same type
        current_activity = None
        current_start = None
        
        for i, time_point in enumerate(unique_times):
            # Get the activity at this time point
            activity_at_time = all_data[all_data['time_of_day'] == time_point]['activity'].iloc[0]
            
            if current_activity is None:
                # First activity
                current_activity = activity_at_time
                current_start = time_point
            elif activity_at_time == current_activity:
                # Same activity continues - do nothing, keep extending
                continue
            else:
                # Activity changed - save the previous range and start new one
                if i < len(unique_times) - 1:
                    # End at the current time point
                    end_time = time_point
                else:
                    # This is the last segment, extend slightly
                    end_time = time_point + 0.01
                
                ranges.append((current_start, end_time, current_activity))
                
                # Start new activity
                current_activity = activity_at_time
                current_start = time_point
        
        # Don't forget the last range
        if current_activity is not None:
            if len(unique_times) > 0:
                end_time = unique_times[-1] + 0.01  # Extend slightly for visibility
                ranges.append((current_start, end_time, current_activity))
        
        # Plot all ranges on a single line
        for start_time, end_time, activity in ranges:
            # Use exact duration - no minimum width to avoid overlaps
            duration = end_time - start_time
            
            # Only plot if duration is positive
            if duration > 0:
                # Check if this activity period is abnormal
                is_abnormal = False
                abnormal_type = None
                
                if abnormal_activities:
                    # Check for unusual timing
                    for abnormal in abnormal_activities.get('unusual_timing', []):
                        if (abnormal['activity'] == activity and 
                            start_time <= abnormal['time_of_day'] <= end_time):
                            is_abnormal = True
                            abnormal_type = 'unusual_timing'
                            break
                    
                    # Check for excessive duration
                    if not is_abnormal:
                        for abnormal in abnormal_activities.get('excessive_duration', []):
                            if (abnormal['activity'] == activity and 
                                abs(duration - abnormal['duration']) < 0.01):  # Within 6 minutes
                                is_abnormal = True
                                abnormal_type = 'excessive_duration'
                                break
                
                # Choose color and edge style based on abnormality
                if is_abnormal:
                    # Use red edge for abnormal activities
                    edge_color = 'red'
                    edge_width = 3
                    # Add a subtle red overlay
                    plt.barh(y_pos, duration, left=start_time, height=0.8, 
                            color='red', alpha=0.3, edgecolor='none')
                else:
                    edge_color = 'black'
                    edge_width = 0.5
                
                plt.barh(y_pos, duration, left=start_time, height=0.8, 
                        color=activity_colors[activity], 
                        edgecolor=edge_color, linewidth=edge_width)
                
                # Add abnormal activity indicator
                if is_abnormal:
                    center_time = start_time + duration / 2
                    plt.text(center_time, y_pos, '!', 
                            ha='center', va='center', fontsize=16, 
                            color='red', fontweight='bold')
        
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
                                       edgecolor='black', label=activity) 
                          for activity in activities]
        
        # Add abnormal activity indicators to legend
        if abnormal_activities and (abnormal_activities.get('unusual_timing') or 
                                   abnormal_activities.get('excessive_duration')):
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor='red', 
                                               alpha=0.3, edgecolor='red', linewidth=3, 
                                               label='Abnormal Activity'))
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='red', 
                                            markerfacecolor='red', markersize=8, 
                                            label='Abnormal Indicator'))
        
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
        
        # Run abnormal activity detection
        if target_date is not None:
            filtered_df = self.filter_by_date(target_date)
            if filtered_df is not None and len(filtered_df) > 0:
                print(f"\nRunning abnormal detection for {target_date}...")
                abnormal_results = self.detect_abnormal_activities(filtered_df)
            else:
                print(f"No data found for abnormal detection on {target_date}")
        else:
            print("\nRunning abnormal detection for all data...")
            abnormal_results = self.detect_abnormal_activities()
        
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