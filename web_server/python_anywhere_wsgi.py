import sys
import os

# Add your project directory to the sys.path
project_home = '/home/chiennd1702/ds/web_server'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Import your Flask application
from server import app as application  # For Flask applications
