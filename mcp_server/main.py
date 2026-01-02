from fastmcp import FastMCP
import requests
import os
import json

mcp = FastMCP("Hamster Monitor")

# Configuration
HARDWARE_SERVER_URL = os.getenv("HARDWARE_SERVER_URL", "http://192.168.50.167:8081")

@mcp.tool()
def get_hamster_status() -> str:
    """
    Get the current status of hamster.
    """
    print("get_hamster_status() called")

    # call the API to get the status of the hamster
    response = requests.get(f"{HARDWARE_SERVER_URL}/status")
    status_report = response.json()

    # status_report = {
    #     "activity": "Sleeping",
    #     "cage_temperature": "21.2°C",
    #     "cage_humidity": "70%",
    #     "cage_ammonia_level": "30ppm",
    #     "weight": "110g",
    #     "sleep_time": "10 hours",
    #     "activity_time": "5 hours",
    #     "heart_rate": "250 bpm",
    #     "body_temperature": "37°C",
    # }
    print("get_hamster_status() result: ", status_report)
    
    # Convert dictionary to formatted string
    return json.dumps(status_report, indent=2)

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(host="0.0.0.0", port=8000, transport="sse")









