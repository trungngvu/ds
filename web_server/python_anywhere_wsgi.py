import sys
import os


project_home = '/home/chiennd1702/ds/web_server'
if project_home not in sys.path:
    sys.path.insert(0, project_home)


from server import app as application  
