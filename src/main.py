"""Main application module for the Hamster Monitor system."""

from src.cameras.camera_manager import CameraManager
from src.sensors.sensor_manager import SensorManager
from src.web.server import WebServer

def main():
    try:
        # Initialize managers
        camera_manager = CameraManager()
        sensor_manager = SensorManager()
        
        # Start camera 0 frame feeding
        camera_manager.start_camera0_feeding()
        
        # Initialize and run web server
        server = WebServer(camera_manager, sensor_manager)
        server.run()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup resources
        if 'camera_manager' in locals():
            camera_manager.cleanup()

if __name__ == '__main__':
    main() 