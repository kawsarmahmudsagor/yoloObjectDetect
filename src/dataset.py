
from roboflow import Roboflow
rf = Roboflow(api_key="V7gbNKr6N2UZG7FvzG6H")
project = rf.workspace("xr23").project("foot-landmark-detection-bhwqm")
version = project.version(5)
dataset = version.download("yolov8")
                
                
                