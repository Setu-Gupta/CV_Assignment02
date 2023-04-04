from skspatial.objects import Plane, Points, Line
import numpy as np
import open3d as o3d
import glob
import csv
import matplotlib.pyplot as plt

# Get the list of all LIDAR scans
pcd_files = glob.glob('./data/lidar_scans/*.pcd')

with open('lidar_normals_and_offsets.csv', 'w', newline='') as csvfile:
    fieldnames = ["Frame", "Normals", "Offset"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for fname in pcd_files:
        # Get the name of the frame
        frame_name = fname.split('.')[-2].split('/')[-1]

        # Read all the coordinates from the PCD file
        pcd = o3d.io.read_point_cloud(fname)
        coords = np.asarray(pcd.points)
        
        # Fit a plane through the coordinates
        points = Points(coords)
        plane = Plane.best_fit(points)
        
        # Get the normal to the plane
        normal = plane.normal

        # Get the magnitude of the offset from the origin (w.r.t. LIDAR coordinate frame)
        offset = np.linalg.norm(plane.point)
        
        row = { "Frame"     : frame_name,
                "Normals"   : normal,
                "Offset"    : offset
              }
        writer.writerow(row)
