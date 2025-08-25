# 🐹 Hamster Activity Charting System

This system provides comprehensive charting and analysis capabilities for your hamster activity monitoring. It automatically logs activity data and generates various types of charts to help you understand your hamster's behavior patterns.

## 📊 Features

-   **Automatic Data Logging**: All activity detections are automatically stored in a SQLite database
-   **Multiple Chart Types**: Time series, distribution, heatmaps, and interactive dashboards
-   **Web Interface**: Beautiful web dashboard for viewing charts and statistics
-   **Data Export**: Export activity data as JSON for further analysis
-   **Real-time Statistics**: Live statistics and activity summaries

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Activity Monitoring

Run your main monitoring system to start collecting data:

```bash
python main.py
```

This will automatically log all activity detections to the database.

### 3. Generate Charts

#### Option A: Standalone Chart Generation

```bash
python generate_charts.py 24  # Generate charts for last 24 hours
```

#### Option B: Web Dashboard

```bash
python chart_web_interface.py
```

Then open http://localhost:8082 in your browser.

## 📈 Chart Types

### 1. Time Series Chart

-   Shows activity probabilities over time
-   Includes motion intensity tracking
-   Helps identify activity patterns and trends

### 2. Activity Distribution

-   Pie chart showing activity frequency
-   Bar chart of average probabilities
-   Overview of most common activities

### 3. Hourly Heatmap

-   Shows activity patterns by hour of day
-   Helps identify daily routines
-   Color-coded intensity levels

### 4. Interactive Dashboard

-   Comprehensive HTML dashboard with multiple charts
-   Zoom, pan, and hover interactions
-   Downloadable for offline viewing

## 🔧 Configuration

### Time Periods

You can analyze data for different time periods:

-   1 hour
-   6 hours
-   12 hours
-   24 hours (default)
-   48 hours
-   1 week (168 hours)

### Chart Customization

The `ActivityCharts` class supports customization:

-   Chart colors
-   Output formats (PNG, HTML)
-   Chart sizes and styling

## 📁 File Structure

```
HamsterMonitor/
├── activity_logger.py      # Data logging and storage
├── activity_charts.py      # Chart generation
├── chart_web_interface.py  # Web dashboard
├── generate_charts.py      # Standalone chart generator
├── main.py                 # Main monitoring system (updated)
├── hamster_activity.db     # SQLite database (auto-created)
├── charts/                 # Generated charts directory
└── requirements.txt        # Python dependencies
```

## 🗄️ Database Schema

The system uses SQLite to store activity data:

```sql
CREATE TABLE activity_logs (
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
);
```

## 📊 API Endpoints

### Web Dashboard API

-   `GET /` - Main dashboard page
-   `GET /api/stats/<hours>` - Get activity statistics
-   `GET /api/generate-charts/<hours>` - Generate charts
-   `GET /api/export-data/<hours>` - Export data as JSON
-   `GET /chart/time-series/<filename>` - Serve time series chart
-   `GET /chart/distribution/<filename>` - Serve distribution chart
-   `GET /chart/heatmap/<filename>` - Serve heatmap chart
-   `GET /chart/dashboard/<filename>` - Serve interactive dashboard

## 🔍 Usage Examples

### Generate Charts for Different Time Periods

```python
from activity_logger import ActivityLogger
from activity_charts import ActivityCharts

# Initialize
logger = ActivityLogger()
charts = ActivityCharts(logger)

# Generate charts for last 6 hours
charts.generate_all_charts(6, "charts")

# Generate charts for last week
charts.generate_all_charts(168, "weekly_charts")
```

### Get Activity Statistics

```python
# Get summary for last 24 hours
summary = logger.get_activity_summary(24)
print(f"Total records: {summary['total_records']}")
print(f"Activities: {summary['activity_counts']}")
print(f"Average motion: {summary['avg_motion']}")
```

### Export Data

```python
# Export last 48 hours of data
filename = logger.export_to_json(48)
print(f"Data exported to: {filename}")
```

## 🎨 Customization

### Custom Colors

You can customize chart colors in `activity_charts.py`:

```python
self.colors = {
    'running': '#FF6B6B',
    'eating': '#4ECDC4',
    'drinking': '#45B7D1',
    'resting': '#96CEB4',
    'exploring': '#FFEAA7'
}
```

### Custom Chart Types

Add new chart types by extending the `ActivityCharts` class:

```python
def create_custom_chart(self, hours: int = 24, save_path: str = None) -> str:
    # Your custom chart implementation
    pass
```

## 🐛 Troubleshooting

### No Data Available

If you see "No data available" messages:

1. Make sure your main monitoring system is running
2. Check that activity detection is working
3. Verify the database file exists: `hamster_activity.db`

### Chart Generation Errors

If charts fail to generate:

1. Install required dependencies: `pip install matplotlib pandas plotly`
2. Check that the `charts/` directory exists
3. Verify you have write permissions

### Web Interface Issues

If the web dashboard doesn't work:

1. Check that port 8082 is available
2. Verify Flask is installed: `pip install flask`
3. Check browser console for JavaScript errors

## 📈 Data Analysis Tips

### Understanding Activity Patterns

-   **High motion + no specific activity** = Exploring
-   **Low motion** = Resting
-   **Consistent patterns** = Daily routines
-   **Sudden changes** = Health or environmental changes

### Best Practices

-   Collect data for at least 24 hours for meaningful patterns
-   Compare different time periods to identify trends
-   Export data regularly for backup
-   Use the interactive dashboard for detailed analysis

## 🔄 Integration

The charting system is automatically integrated with your existing monitoring system. When you run `main.py`, it will:

1. Detect hamster activities
2. Log all data to the database
3. Provide real-time activity information
4. Enable chart generation from collected data

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify all dependencies are installed
3. Check the database for data: `sqlite3 hamster_activity.db "SELECT COUNT(*) FROM activity_logs;"`
4. Review the console output for error messages

---

**Happy Hamster Monitoring! 🐹📊**
