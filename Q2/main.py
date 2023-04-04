from skspatial.objects import Plane, Points, Line
from skspatial.plotting import plot_3d
import numpy as np
import open3d as o3d
import glob
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimage

# ====================================================== Part 1 =========================================================
# Create a dictionary to store the normals and the offsets
lidar_extrinsics = {}

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

        # Save the data lidar_extrinsics
        lidar_extrinsics[frame_name] = [normal, np.asarray(plane.point)]

# ====================================================== Part 3 =========================================================
# Compute the change of basis transformation from LIDAR coordinates to checkerboard coordinates
def get_transform(camera_rotation, camera_translation, lidar_normal, lidar_offset):
    
    # Get the translation vector from LIDAR coordinates to checkerboard coordinates
    ch_t_L = -1 * np.asarray(lidar_offset).reshape(3, 1)
    z_L = np.array([0, 0, 1])
    z_ch = lidar_normal
    
    # Compute the cosine of the angle between z_l and z_ch
    cos_theta = z_L.dot(z_ch)

    # Compute the normalized axis of rotation
    a = np.cross(z_L, z_ch)
    a = a/np.linalg.norm(a)

    # Build the rotation matrix ch_R_L
    A = np.cross(a, np.identity(3) * -1)
    I = np.identity(3)
    ch_R_L = I + (np.sqrt(1 - np.power(cos_theta, 2)) * A) + ((1 - cos_theta) * A * A)
    det = np.linalg.det(ch_R_L)
    ch_R_L = ch_R_L * (1 / det) # Force the determinant to be one

    # Build the change of basis matrix ch_B_L
    rt = ch_R_L.dot(ch_t_L)
    ch_B_L = np.append(ch_R_L, rt, axis=1)
    bottom_row = np.asarray([0., 0., 0., 1.]).reshape(1, 4)
    ch_B_L = np.append(ch_B_L, bottom_row, axis=0)

    # Get the camera rotation matrix and the translation vector
    C_R_ch = np.asarray(camera_rotation)
    C_t_ch = np.asarray(camera_translation).reshape(3, 1)

    # Build the change of basis matrix C_B_ch
    C_B_ch = np.append(C_R_ch, C_t_ch, axis=1)
    bottom_row = np.asarray([0., 0., 0., 1.]).reshape(1, 4)
    C_B_ch = np.append(C_B_ch, bottom_row, axis=0)

    # Compute the change of basis matrix C_B_L
    C_B_L =  C_B_ch * ch_B_L

    # Compute the rotation matrix
    C_R_L = C_R_ch.dot(ch_R_L)
    
    return C_B_L, C_R_L

# ====================================================== Part 4 =========================================================
rotation_matrices = {}
for fname in pcd_files:
    # Get the name of the frame
    frame_name = fname.split('.')[-2].split('/')[-1]
    
    # Get the LIDAR parameters
    lidar_normal = lidar_extrinsics[frame_name][0]
    lidar_offset = lidar_extrinsics[frame_name][1]
    
    # Get the camera parameters path
    parameters_path = fname.replace('lidar_scans', 'camera_parameters').replace('pcd', 'jpeg')
    rotation_path = parameters_path + '/rotation_matrix.txt'
    translation_path = parameters_path + '/translation_vectors.txt'
    
    # Get the camera rotation matrix and the translation vector
    t = []
    with open(translation_path, 'r') as tfile:
        for line in tfile.readlines():
            t.append(float(line))

    r = []
    with open(rotation_path, 'r') as rotfile:
        for line in rotfile.readlines():
            row = [float(x) for x in line.split()]
            r.append(row)
    
    camera_rotation = np.asarray(r).reshape(3, 3)
    camera_translation = np.asarray(t).reshape(3, 1)

    # Get the transformation matrix
    C_B_L, C_R_L = get_transform(camera_rotation, camera_translation, lidar_normal, lidar_offset)
    rotation_matrices[frame_name] = C_R_L

    # Get the camera instrinsic matrix path
    intrinsic_path = './data/camera_parameters/camera_intrinsic.txt'

    # Get the instrinsic matrix
    int_mat = []
    with open(intrinsic_path, 'r') as intfile:
        for line in intfile.readlines():
            row = [float(x) for x in line.split()]
            int_mat.append(row)
    
    Int = np.asarray(int_mat).reshape(3, 3)

    # Compute the perspective projection matrix
    I = np.identity(3)
    zeros = np.zeros(3).reshape(3, 1)
    Pro = np.append(I, zeros, axis=1)
    
    # Compute the image formation pipeline matrix
    F = Int.dot(Pro.dot(C_B_L))
    
    # Get the image path
    image_path = fname.replace('lidar_scans', 'camera_images').replace('pcd', 'jpeg')

    # Plot the image
    plt.figure()
    img = mpimage.imread(image_path)
    plt.imshow(img)
    
    # Read all the coordinates from the PCD file
    pcd = o3d.io.read_point_cloud(fname)
    coords = np.asarray(pcd.points)
     
    # Go over every point in the PCD file
    for point in coords:
        homo_point = np.append(np.asarray(point), np.asarray([1])).reshape(4, 1)
         
        # Get the projection of the point in the image frame
        point_image_frame = F.dot(homo_point)
        
        # Get the pixel coordinates by dividing by the homogeneous coordinate
        point_image_frame = point_image_frame / point_image_frame[2]
    
        # Change to pixel coordinates by casting into integer
        point_pixel_coords = point_image_frame[:-1].astype('int')

        # Plot the point onto the image
        plt.plot(point_pixel_coords[0], point_pixel_coords[1], color='r', marker='+', markersize=5)
    
    # Save the images with LIDAR points mapped onto them
    image_name = fname.split('/')[-1].replace('pcd', 'jpeg')
    save_path = './mappings/' + image_name
    plt.savefig(save_path)
    plt.close()

# ====================================================== Part 5 =========================================================
with open('cos_dist.csv', 'w', newline='') as csvfile:
    fieldnames = ["Frame", "Cosine Distance"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    cosine_dist = []
    for fname in pcd_files:
        # Get the name of the frame
        frame_name = fname.split('.')[-2].split('/')[-1]
        
        # Get the LIDAR parameters
        lidar_normal = lidar_extrinsics[frame_name][0]

        # Get the camera normal flie path
        camera_normal_path = './data/camera_parameters/' + frame_name + '.jpeg/camera_normals.txt'
        
        # Read the camera normal
        camera_normal = []
        with open(camera_normal_path, 'r') as normalfile:
            for line in normalfile.readlines():
                camera_normal.append(float(line))
        camera_normal = np.asarray(camera_normal)

        # Get the rotation matrix
        C_R_L = rotation_matrices[frame_name]

        # Compute the rotated LIDAR normal
        rotated_lidar_normal = C_R_L.dot(lidar_normal)
        
        # Create 3 lines, one for each normal
        line_camera_normal = Line(point=[0, 0, 0], direction=camera_normal)
        line_lidar_normal = Line(point=[0, 0, 0], direction=lidar_normal)
        line_rotated_normal = Line(point=[0, 0, 0], direction=rotated_lidar_normal)

        # Plot the lines
        plt.figure()
        plot_3d(line_camera_normal.plotter(c='r', label='Camera Normal'),
            line_lidar_normal.plotter(c='g', label='LIDAR Normal'),
            line_rotated_normal.plotter(c='b', label='Rotated LIDAR Normal')
        )
        plt.legend()
        image_path = './normal_plots/' + frame_name + '.png' 
        plt.savefig(image_path)
        plt.close()

        # Compute the cosine distance
        dist = camera_normal.dot(rotated_lidar_normal) / (np.linalg.norm(rotated_lidar_normal) * np.linalg.norm(camera_normal))

        # Save the distance in the CSV file
        row = {'Frame'              : frame_name,
               'Cosine Distance'    : dist
              }
        writer.writerow(row)

        # Save the distance for plotting the histogram
        cosine_dist.append(dist)
    
    # Compute the average and the standard deviation
    mean = np.mean(cosine_dist)
    std = np.std(cosine_dist)
    print("Average error:", mean)
    print("Standard deviation of error:", std)

    # Plot the histogram of distances
    plt.figure()
    image_path = './dist.png'
    plt.hist(cosine_dist)
    plt.title('Histogram of Cosine Distances')
    plt.xlabel('Cosine Distance')
    plt.ylabel('Count')
    plt.savefig(image_path)
    plt.close()
