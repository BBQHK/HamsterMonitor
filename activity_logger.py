import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import sqlite3
from pathlib import Path

class ActivityLogger:
    def __init__(self, db_path: str = "hamster_activity.db"):
        """
        Initialize the activity logger.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create activity_logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                activity TEXT NOT NULL,
                probability REAL NOT NULL,
                running_prob REAL DEFAULT 0.0,
                eating_prob REAL DEFAULT 0.0,
                drinking_prob REAL DEFAULT 0.0,
                resting_prob REAL DEFAULT 0.0,
                exploring_prob REAL DEFAULT 0.0,
                motion_intensity REAL DEFAULT 0.0
            )
        ''')
        
        # Create index for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON activity_logs(timestamp)
        ''')
        
        conn.commit()
        conn.close()
    
    def log_activity(self, activity: str, activity_probs: Dict[str, float], 
                    motion_intensity: float = 0.0):
        """
        Log activity data to the database.
        
        Args:
            activity: Detected activity
            activity_probs: Dictionary of activity probabilities
            motion_intensity: Motion intensity value
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO activity_logs 
            (timestamp, activity, probability, running_prob, eating_prob, 
             drinking_prob, resting_prob, exploring_prob, motion_intensity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            activity,
            activity_probs.get(activity, 0.0),
            activity_probs.get('running', 0.0),
            activity_probs.get('eating', 0.0),
            activity_probs.get('drinking', 0.0),
            activity_probs.get('resting', 0.0),
            activity_probs.get('exploring', 0.0),
            motion_intensity
        ))
        
        conn.commit()
        conn.close()
    
    def get_activity_data(self, hours: int = 24) -> List[Dict]:
        """
        Get activity data for the specified time period.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of activity records
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, activity, probability, running_prob, eating_prob,
                   drinking_prob, resting_prob, exploring_prob, motion_intensity
            FROM activity_logs
            WHERE timestamp >= datetime('now', '-{} hours')
            ORDER BY timestamp
        '''.format(hours))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'timestamp': row[0],
                'activity': row[1],
                'probability': row[2],
                'running_prob': row[3],
                'eating_prob': row[4],
                'drinking_prob': row[5],
                'resting_prob': row[6],
                'exploring_prob': row[7],
                'motion_intensity': row[8]
            }
            for row in rows
        ]
    
    def get_activity_summary(self, hours: int = 24) -> Dict:
        """
        Get activity summary statistics.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Dictionary with activity statistics
        """
        data = self.get_activity_data(hours)
        
        if not data:
            return {
                'total_records': 0,
                'activity_counts': {},
                'avg_probabilities': {},
                'avg_motion': 0.0
            }
        
        # Count activities
        activity_counts = {}
        total_probabilities = {
            'running': 0.0,
            'eating': 0.0,
            'drinking': 0.0,
            'resting': 0.0,
            'exploring': 0.0
        }
        total_motion = 0.0
        
        for record in data:
            activity = record['activity']
            activity_counts[activity] = activity_counts.get(activity, 0) + 1
            
            total_probabilities['running'] += record['running_prob']
            total_probabilities['eating'] += record['eating_prob']
            total_probabilities['drinking'] += record['drinking_prob']
            total_probabilities['resting'] += record['resting_prob']
            total_probabilities['exploring'] += record['exploring_prob']
            total_motion += record['motion_intensity']
        
        # Calculate averages
        num_records = len(data)
        avg_probabilities = {
            k: v / num_records for k, v in total_probabilities.items()
        }
        avg_motion = total_motion / num_records
        
        return {
            'total_records': num_records,
            'activity_counts': activity_counts,
            'avg_probabilities': avg_probabilities,
            'avg_motion': avg_motion,
            'time_period_hours': hours
        }
    
    def export_to_json(self, hours: int = 24, filename: str = None) -> str:
        """
        Export activity data to JSON file.
        
        Args:
            hours: Number of hours to export
            filename: Output filename (optional)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = f"hamster_activity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = self.get_activity_data(hours)
        summary = self.get_activity_summary(hours)
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'time_period_hours': hours,
            'summary': summary,
            'detailed_data': data
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filename 