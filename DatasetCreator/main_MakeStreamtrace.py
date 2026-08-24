import os
import glob
from paraview.simple import *

# pvpython make_streamtracer.py 

def config_camera(view):
    # CustomViewpointButton4 (Viewpoint 5)
    view.CameraPosition = [-145.62500334229827, 258.12891252294224, 337.27420872575703]
    view.CameraFocalPoint = [54.23265623503145, 49.59503372400067, 63.19529352193284]
    view.CameraViewUp = [0.28077913995658044, 0.8513959344245512, -0.4430440580919559]
    view.CameraViewAngle = 30
    view.CameraParallelScale = 103.05702305034819
    view.CenterOfRotation = [59.5, 59.5, 59.5]

if __name__ == "__main__":
    
    # --- MAIN SETTINGS ---
    BASE_FOLDER_PATH = "../../GradSimulations/" 
    # Set to None to save next to the .vti, or a string (e.g., "streamtrace_images")
    EXPERIMENT_NAME = None 
    SOLID_COLOR = [0.5, 0.5, 0.5] 
    
    # Toggle the color bar on (True) or off (False)
    SHOW_COLORBAR = True 
    # ---------------------

    # Only create a global output directory if EXPERIMENT_NAME is defined
    if EXPERIMENT_NAME is not None:
        OUT_DIR = os.path.join(BASE_FOLDER_PATH, EXPERIMENT_NAME)
        if not os.path.exists(OUT_DIR):
            os.makedirs(OUT_DIR)

    # 0) Gather all the target files first (Recursive search)
    target_files = []
    
    # os.walk goes forward into every single nested subfolder automatically
    for root, dirs, files in os.walk(BASE_FOLDER_PATH):
        
        # Skip the global output folder so it doesn't search inside it (if it exists)
        if EXPERIMENT_NAME is not None and EXPERIMENT_NAME in root:
            continue
            
        # Look for .vti files in the current subfolder being crawled
        search_pattern = os.path.join(root, "output_*.vti")
        vti_files = sorted(glob.glob(search_pattern))
        
        if vti_files:
            target_files.append(vti_files[-1]) # Grab the highest file in this specific folder

    if not target_files:
        print("No VTI files found. Exiting.")
        exit()

    print(f"Found {len(target_files)} files. Setting up the Ray Tracing pipeline once...")

    # --- SETUP PIPELINE ONCE ---
    ResetSession()

    # Load the first file just to initialize the structures
    reader = XMLImageDataReader(FileName=[target_files[0]])
    reader.PointArrayStatus = ['Density', 'Velocity']
    reader.UpdatePipeline()

    view = CreateRenderView()
    view.ViewSize = [1200, 1200]
    view.Background = [1.0, 1.0, 1.0] 
    view.UseColorPaletteForBackground = 0
    view.OrientationAxesVisibility = 0

    # --- Rendering Aspects (OSPRay Ray Tracing) ---
    view.EnableRayTracing = 1
    view.Shadows = 0                
    view.SamplesPerPixel = 40       
    view.AmbientSamples = 2         
    if hasattr(view, "EnableOSPRayDenoiser"):
        view.EnableOSPRayDenoiser = 1

    # Filters
    thresh = Threshold(Input=reader)
    thresh.Scalars = ['POINTS', 'Density']
    thresh.ThresholdMethod = 'Between'
    thresh.LowerThreshold = -1e10  
    thresh.UpperThreshold = 0.0

    # 1. Main Solid Geometry (Shows 7/8ths of the geometry)
    clip1 = Clip(Input=thresh)
    clip1.ClipType = 'Plane'
    clip1.ClipType.Normal = [-1.0, 0.0, 0.0]
    clip1.ClipType.Origin = [59.5, 59.5, 59.5] 

    clip2 = Clip(Input=thresh)
    clip2.ClipType = 'Plane'
    clip2.ClipType.Normal = [0.0, 1.0, 0.0]
    clip2.ClipType.Origin = [59.5, 59.5, 59.5]
    
    clip3 = Clip(Input=thresh)
    clip3.ClipType = 'Plane'
    clip3.ClipType.Normal = [0.0, 0.0, 1.0] 
    clip3.ClipType.Origin = [59.5, 59.5, 59.5]
    
    # --- HIGHLIGHT FRAMES (Manual Custom Wireframe) ---
    # We explicitly define the 21 line segments that frame the geometry
    # to perfectly recreate the cut corner without lines floating in the void.
    wireframe_coords = [
        # --- 1. THE 9 INNER STEP LINES ---
        # Inner creases radiating from the center vertex (59.5, 59.5, 59.5)
        ([59.5, 59.5, 59.5], [0.0, 59.5, 59.5]),
        ([59.5, 59.5, 59.5], [59.5, 119.0, 59.5]),
        ([59.5, 59.5, 59.5], [59.5, 59.5, 119.0]),
        # Face lines on the X=0 boundary
        ([0.0, 59.5, 59.5], [0.0, 119.0, 59.5]),
        ([0.0, 59.5, 59.5], [0.0, 59.5, 119.0]),
        # Face lines on the Y=119 boundary
        ([59.5, 119.0, 59.5], [0.0, 119.0, 59.5]),
        ([59.5, 119.0, 59.5], [59.5, 119.0, 119.0]),
        # Face lines on the Z=119 boundary
        ([59.5, 59.5, 119.0], [0.0, 59.5, 119.0]),
        ([59.5, 59.5, 119.0], [59.5, 119.0, 119.0]),

        # --- 2. THE 9 FULL OUTER DOMAIN LINES ---
        # X-aligned edges
        ([0.0, 0.0, 0.0], [119.0, 0.0, 0.0]),
        ([0.0, 119.0, 0.0], [119.0, 119.0, 0.0]),
        ([0.0, 0.0, 119.0], [119.0, 0.0, 119.0]),
        # Y-aligned edges
        ([0.0, 0.0, 0.0], [0.0, 119.0, 0.0]),
        ([119.0, 0.0, 0.0], [119.0, 119.0, 0.0]),
        ([119.0, 0.0, 119.0], [119.0, 119.0, 119.0]),
        # Z-aligned edges
        ([0.0, 0.0, 0.0], [0.0, 0.0, 119.0]),
        ([119.0, 0.0, 0.0], [119.0, 0.0, 119.0]),
        ([119.0, 119.0, 0.0], [119.0, 119.0, 119.0]),

        # --- 3. THE 3 TRUNCATED OUTER DOMAIN LINES ---
        # These stop exactly where the cut begins, leaving the void corner empty.
        ([0.0, 119.0, 0.0], [0.0, 119.0, 59.5]),       # Z-aligned, stops at Z=59.5
        ([119.0, 119.0, 119.0], [59.5, 119.0, 119.0]), # X-aligned, stops at X=59.5
        ([0.0, 0.0, 119.0], [0.0, 59.5, 119.0])        # Y-aligned, stops at Y=59.5
    ]

    # Convert the mathematical coordinates into 3D tubes
    wireframe_tubes = []
    for pt1, pt2 in wireframe_coords:
        l = Line(Point1=pt1, Point2=pt2)
        t = Tube(Input=l)
        t.Radius = 0.3
        t.Capping = 1
        wireframe_tubes.append(t)
    # -------------------------------------------------

    # 3. Streamlines
    stream = StreamTracer(Input=reader, SeedType='Point Cloud')
    stream.Vectors = ['POINTS', 'Velocity']
    stream.MaximumStreamlineLength = 1000.0 
    stream.SeedType.Center = [59.5, 59.5, 59.5]
    stream.SeedType.Radius = 120.0
    stream.SeedType.NumberOfPoints = 1150

    # --- DISPLAY SETTINGS ---
    # Show Solid Geometry
    disp_clip1 = Show(clip1, view)
    disp_clip1.ColorArrayName = ['POINTS', '']  
    disp_clip1.DiffuseColor = SOLID_COLOR
    disp_clip1.AmbientColor = SOLID_COLOR
    disp_clip1.Opacity = 1.0

    disp_clip2 = Show(clip2, view)
    disp_clip2.ColorArrayName = ['POINTS', '']  
    disp_clip2.DiffuseColor = SOLID_COLOR
    disp_clip2.AmbientColor = SOLID_COLOR
    disp_clip2.Opacity = 1.0
    
    disp_clip3 = Show(clip3, view)
    disp_clip3.ColorArrayName = ['POINTS', '']  
    disp_clip3.DiffuseColor = SOLID_COLOR
    disp_clip3.AmbientColor = SOLID_COLOR
    disp_clip3.Opacity = 1.0
    
    # Show Custom Wireframe
    for t in wireframe_tubes:
        disp_frame = Show(t, view)
        disp_frame.ColorArrayName = ['POINTS', '']  
        disp_frame.DiffuseColor = [0.0, 0.0, 0.0]  
        disp_frame.AmbientColor = [0.0, 0.0, 0.0]

    # Show Streamlines
    disp_stream = Show(stream, view)
    ColorBy(disp_stream, ('POINTS', 'Velocity', 'Magnitude'))
    
    velocityLUT = GetColorTransferFunction('Velocity')
    velocityLUT.ApplyPreset('autumn (matplotlib)', True) 
    
    # Color bar visibility and text formatting
    if SHOW_COLORBAR:
        disp_stream.SetScalarBarVisibility(view, True)
        
        color_bar = GetScalarBar(velocityLUT, view)
        
        # Force the title and numbers to be black and serif
        color_bar.TitleColor = [0.0, 0.0, 0.0]
        color_bar.LabelColor = [0.0, 0.0, 0.0]
        color_bar.TitleFontFamily = 'Times'
        color_bar.LabelFontFamily = 'Times'
        color_bar.TitleFontSize = 28
        color_bar.LabelFontSize = 28
        color_bar.AutomaticLabelFormat = 0
        color_bar.LabelFormat = '%.2e'
    else:
        disp_stream.SetScalarBarVisibility(view, False)

    config_camera(view)
    
    # Do a dummy render to force OSPRay and the Denoiser to compile
    print("Compiling shaders and priming Denoiser...")
    Render() 

    # --- LOOP THROUGH FILES ---
    for file_path in target_files:
        print(f"Processing: {file_path}...")
        
        # 1. Swap out the file name in the reader
        reader.FileName = [file_path]
        reader.UpdatePipeline()
        
        # 2. Determine save location
        parent_folder_name = os.path.basename(os.path.dirname(file_path))
        out_filename = f"streamlines_{parent_folder_name}.png"
        
        if EXPERIMENT_NAME is None:
            output = os.path.join(os.path.dirname(file_path), out_filename)
        else:
            output = os.path.join(OUT_DIR, out_filename)
        
        # 3. Render and save the image
        Render()
        SaveScreenshot(output, view, ImageResolution=[1200, 1200], TransparentBackground=0)
        
    print("Finished processing all folders!")