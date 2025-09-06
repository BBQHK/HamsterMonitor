import sqlite3
import os
from datetime import datetime
from typing import Dict, Any

class HamsterDatabase:
    def __init__(self, db_path: str = "hamster_activity.db"):
        """Initialize the database connection and create tables if they don't exist."""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Create the database and tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create activity_logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                activity TEXT NOT NULL,
                confidence REAL NOT NULL,
                running_prob REAL,
                eating_prob REAL,
                drinking_prob REAL,
                resting_prob REAL,
                exploring_prob REAL,
                motion_intensity REAL,
                frame_width INTEGER,
                frame_height INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_activity(self, activity: str, confidence: float, all_probabilities: Dict[str, float], 
                    motion_intensity: float = None, frame_shape: tuple = None):
        """Log a single activity detection to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get frame dimensions if provided
        frame_width = frame_shape[1] if frame_shape and len(frame_shape) >= 2 else None
        frame_height = frame_shape[0] if frame_shape and len(frame_shape) >= 1 else None
        
        cursor.execute('''
            INSERT INTO activity_logs 
            (timestamp, activity, confidence, running_prob, eating_prob, drinking_prob, 
             resting_prob, exploring_prob, motion_intensity, frame_width, frame_height)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(),
            activity,
            confidence,
            all_probabilities.get('running', 0.0),
            all_probabilities.get('eating', 0.0),
            all_probabilities.get('drinking', 0.0),
            all_probabilities.get('resting', 0.0),
            all_probabilities.get('exploring', 0.0),
            motion_intensity,
            frame_width,
            frame_height
        ))
        
        conn.commit()
        conn.close()
    
    def close(self):
        """Close the database connection."""
        pass  # SQLite connections are automatically closed
