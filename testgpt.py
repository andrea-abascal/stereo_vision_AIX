import open3d as o3d
import numpy as np

# Create a visualization window
vis = o3d.visualization.Visualizer()
vis.create_window()

# Create an empty point cloud object
point_cloud = o3d.geometry.PointCloud()

# Start the video capture from stereo vision
# Replace this section with your stereo vision setup and point cloud generation
# The variable `new_points` should contain the 3D points obtained from the stereo vision
new_points = np.random.rand(1000, 3)  # Example: Generate random points for testing

# Add the new points to the point cloud
point_cloud.points = o3d.utility.Vector3dVector(new_points)

# Update the visualization
vis.add_geometry(point_cloud)
vis.update_geometry()
vis.poll_events()
vis.update_renderer()

while True:
    # Continue capturing new frames from stereo vision and updating the point cloud

    # Replace this section with your stereo vision setup and point cloud generation
    # The variable `new_points` should contain the updated 3D points
    new_points = np.random.rand(1000, 3)  # Example: Generate random points for testing

    # Update the point cloud with the new points
    point_cloud.points = o3d.utility.Vector3dVector(new_points)

    # Update the visualization
    vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()

# Close the visualization window
vis.destroy_window()
