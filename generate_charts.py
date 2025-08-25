#!/usr/bin/env python3
"""
Hamster Activity Chart Generator

This script demonstrates how to generate various charts for hamster activity analysis.
You can run this script independently to create charts from existing data.

Usage:
    python generate_charts.py [hours]

Example:
    python generate_charts.py 24  # Generate charts for last 24 hours
"""

import sys
import os
from datetime import datetime
from activity_logger import ActivityLogger
from activity_charts import ActivityCharts

def main():
    # Get time period from command line argument
    hours = 24  # Default to 24 hours
    if len(sys.argv) > 1:
        try:
            hours = int(sys.argv[1])
        except ValueError:
            print("Error: Hours must be a number")
            sys.exit(1)
    
    print(f"🐹 Hamster Activity Chart Generator")
    print(f"Generating charts for the last {hours} hours...")
    print("-" * 50)
    
    # Initialize logger and charts
    logger = ActivityLogger()
    charts = ActivityCharts(logger)
    
    # Check if we have data
    summary = logger.get_activity_summary(hours)
    if summary['total_records'] == 0:
        print("❌ No activity data found for the specified time period.")
        print("Make sure your hamster monitoring system has been running and collecting data.")
        print("\nTo collect data:")
        print("1. Run your main.py to start the activity detection")
        print("2. Let it run for a while to collect activity data")
        print("3. Then run this script again")
        return
    
    print(f"✅ Found {summary['total_records']} activity records")
    print(f"📊 Activities detected: {list(summary['activity_counts'].keys())}")
    print(f"📈 Average motion intensity: {summary['avg_motion']:.3f}")
    print()
    
    # Generate charts
    print("🔄 Generating charts...")
    try:
        charts_dict = charts.generate_all_charts(hours, "charts")
        
        print("✅ Charts generated successfully!")
        print("\n📁 Generated files:")
        for chart_type, filepath in charts_dict.items():
            if filepath:
                filename = os.path.basename(filepath)
                print(f"   • {chart_type}: {filename}")
        
        print(f"\n📂 Charts saved in: {os.path.abspath('charts')}")
        print("\n🌐 To view charts in a web interface:")
        print("   python chart_web_interface.py")
        print("   Then open http://localhost:8082 in your browser")
        
    except Exception as e:
        print(f"❌ Error generating charts: {e}")
        print("\nMake sure you have the required dependencies installed:")
        print("   pip install matplotlib pandas plotly")

if __name__ == "__main__":
    main() 