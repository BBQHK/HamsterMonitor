from flask import Flask, render_template_string, jsonify, request, send_file
import os
from datetime import datetime, timedelta
from activity_logger import ActivityLogger
from activity_charts import ActivityCharts
import json

app = Flask(__name__)

# Initialize logger and charts
logger = ActivityLogger()
charts = ActivityCharts(logger)

# HTML template for the dashboard
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hamster Activity Dashboard</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        .controls {
            padding: 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
        }
        .control-group {
            display: inline-block;
            margin-right: 20px;
        }
        .control-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #495057;
        }
        .control-group select, .control-group button {
            padding: 8px 12px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            font-size: 14px;
        }
        .control-group button {
            background: #007bff;
            color: white;
            border: none;
            cursor: pointer;
            transition: background 0.3s;
        }
        .control-group button:hover {
            background: #0056b3;
        }
        .content {
            padding: 20px;
        }
        .chart-section {
            margin-bottom: 30px;
        }
        .chart-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 5px;
        }
        .chart-container {
            text-align: center;
            margin: 20px 0;
        }
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }
        .stat-label {
            color: #6c757d;
            margin-top: 5px;
        }
        .loading {
            text-align: center;
            padding: 40px;
            color: #6c757d;
        }
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 4px;
            margin: 20px 0;
        }
        .export-section {
            background: #e9ecef;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .export-section h3 {
            margin-top: 0;
            color: #495057;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🐹 Hamster Activity Dashboard</h1>
            <p>Real-time activity analysis and visualization</p>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label for="timeRange">Time Range:</label>
                <select id="timeRange">
                    <option value="1">Last Hour</option>
                    <option value="6">Last 6 Hours</option>
                    <option value="12">Last 12 Hours</option>
                    <option value="24" selected>Last 24 Hours</option>
                    <option value="48">Last 48 Hours</option>
                    <option value="168">Last Week</option>
                </select>
            </div>
            <div class="control-group">
                <label>&nbsp;</label>
                <button onclick="generateCharts()">Generate Charts</button>
            </div>
            <div class="control-group">
                <label>&nbsp;</label>
                <button onclick="exportData()">Export Data</button>
            </div>
        </div>
        
        <div class="content">
            <div id="stats" class="stats-grid"></div>
            <div id="charts"></div>
        </div>
    </div>

    <script>
        // Generate charts on page load
        window.onload = function() {
            generateCharts();
        };

        function generateCharts() {
            const hours = document.getElementById('timeRange').value;
            const chartsDiv = document.getElementById('charts');
            const statsDiv = document.getElementById('stats');
            
            // Show loading
            chartsDiv.innerHTML = '<div class="loading">Generating charts...</div>';
            statsDiv.innerHTML = '<div class="loading">Loading statistics...</div>';
            
            // Fetch statistics
            fetch(`/api/stats/${hours}`)
                .then(response => response.json())
                .then(data => {
                    displayStats(data);
                })
                .catch(error => {
                    statsDiv.innerHTML = '<div class="error">Error loading statistics: ' + error.message + '</div>';
                });
            
            // Generate charts
            fetch(`/api/generate-charts/${hours}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        displayCharts(data.charts);
                    } else {
                        chartsDiv.innerHTML = '<div class="error">Error generating charts: ' + data.error + '</div>';
                    }
                })
                .catch(error => {
                    chartsDiv.innerHTML = '<div class="error">Error generating charts: ' + error.message + '</div>';
                });
        }

        function displayStats(stats) {
            const statsDiv = document.getElementById('stats');
            statsDiv.innerHTML = `
                <div class="stat-card">
                    <div class="stat-value">${stats.total_records}</div>
                    <div class="stat-label">Total Records</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.avg_motion.toFixed(3)}</div>
                    <div class="stat-label">Avg Motion</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${Object.keys(stats.activity_counts).length}</div>
                    <div class="stat-label">Activities Detected</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.time_period_hours}h</div>
                    <div class="stat-label">Time Period</div>
                </div>
            `;
        }

        function displayCharts(charts) {
            const chartsDiv = document.getElementById('charts');
            let html = '';
            
            if (charts.time_series) {
                html += `
                    <div class="chart-section">
                        <div class="chart-title">Activity Time Series</div>
                        <div class="chart-container">
                            <img src="/chart/time-series/${charts.time_series}" alt="Time Series Chart">
                        </div>
                    </div>
                `;
            }
            
            if (charts.distribution) {
                html += `
                    <div class="chart-section">
                        <div class="chart-title">Activity Distribution</div>
                        <div class="chart-container">
                            <img src="/chart/distribution/${charts.distribution}" alt="Distribution Chart">
                        </div>
                    </div>
                `;
            }
            
            if (charts.heatmap) {
                html += `
                    <div class="chart-section">
                        <div class="chart-title">Hourly Activity Heatmap</div>
                        <div class="chart-container">
                            <img src="/chart/heatmap/${charts.heatmap}" alt="Heatmap Chart">
                        </div>
                    </div>
                `;
            }
            
            if (charts.dashboard) {
                html += `
                    <div class="chart-section">
                        <div class="chart-title">Interactive Dashboard</div>
                        <div class="export-section">
                            <h3>Interactive Dashboard Available</h3>
                            <p>Download the interactive HTML dashboard for detailed analysis:</p>
                            <a href="/chart/dashboard/${charts.dashboard}" class="control-group button" download>
                                Download Interactive Dashboard
                            </a>
                        </div>
                    </div>
                `;
            }
            
            if (!html) {
                html = '<div class="error">No charts generated. Please check if there is activity data available.</div>';
            }
            
            chartsDiv.innerHTML = html;
        }

        function exportData() {
            const hours = document.getElementById('timeRange').value;
            window.open(`/api/export-data/${hours}`, '_blank');
        }
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """Main dashboard page."""
    return render_template_string(DASHBOARD_TEMPLATE)

@app.route('/api/stats/<int:hours>')
def get_stats(hours):
    """Get activity statistics for the specified time period."""
    try:
        summary = logger.get_activity_summary(hours)
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-charts/<int:hours>')
def generate_charts(hours):
    """Generate charts for the specified time period."""
    try:
        charts_dict = charts.generate_all_charts(hours)
        return jsonify({'success': True, 'charts': charts_dict})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/chart/time-series/<path:filename>')
def serve_time_series_chart(filename):
    """Serve time series chart image."""
    try:
        return send_file(f"charts/{filename}", mimetype='image/png')
    except FileNotFoundError:
        return "Chart not found", 404

@app.route('/chart/distribution/<path:filename>')
def serve_distribution_chart(filename):
    """Serve distribution chart image."""
    try:
        return send_file(f"charts/{filename}", mimetype='image/png')
    except FileNotFoundError:
        return "Chart not found", 404

@app.route('/chart/heatmap/<path:filename>')
def serve_heatmap_chart(filename):
    """Serve heatmap chart image."""
    try:
        return send_file(f"charts/{filename}", mimetype='image/png')
    except FileNotFoundError:
        return "Chart not found", 404

@app.route('/chart/dashboard/<path:filename>')
def serve_dashboard_file(filename):
    """Serve interactive dashboard HTML file."""
    try:
        return send_file(f"charts/{filename}", mimetype='text/html')
    except FileNotFoundError:
        return "Dashboard not found", 404

@app.route('/api/export-data/<int:hours>')
def export_data(hours):
    """Export activity data as JSON."""
    try:
        filename = logger.export_to_json(hours)
        return send_file(filename, mimetype='application/json', as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082, debug=True) 