import open3d as o3d

# Load point cloud data (replace this with your own point cloud)
pcd = o3d.io.read_point_cloud("results/s1pointcloud.ply")

# Create a visualization window
vis = o3d.visualization.VisualizerWithKeyCallback()

# Function to capture the screen and save as PNG
def capture_screen(vis):
    image = vis.capture_screen_image("point_cloudPPT.png",True)
    #o3d.io.write_image("point_cloud1.png", image)

# Register the capture_screen function as the key callback
vis.register_key_callback(ord("S"), capture_screen)

# Add the point cloud to the visualization
vis.create_window('Point Cloud Scene')

vis.add_geometry(pcd)
view_control = vis.get_view_control()
view_control.set_zoom(0.1)
view_control.set_up([ -0.0023108895800350508, -0.98944745178302018, 0.14487373795632214 ])
view_control.set_lookat([53.331142545750687, 26.46709928743525, 68.584828769678921 ])
view_control.set_front([-0.031372204363311264, -0.14487454624381371, -0.98895255227135936])
# Start the visualization loop
vis.run()
vis.destroy_window()

