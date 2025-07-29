import numpy as np
import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf
from plyfile import PlyData
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageFilter
import sys
import os
import scicomap as sc

def load_ply(filepath):
    """Load PLY file with xyz coordinates, pred, pwood, and reflectance columns using plyfile"""
    print(f"Loading PLY file: {filepath}")
    try:
        plydata = PlyData.read(filepath)
        vertex = plydata['vertex']
        
        # Extract coordinates and properties
        x = vertex['x']
        y = vertex['y']
        z = vertex['z']
        
        # Check for property names
        properties = [p.name for p in vertex.properties]
        print(f"Available properties: {properties}")
        
        # Create a case-insensitive mapping to standardize property names (removing 'scalar_' prefix)
        property_map = {}
        for prop in properties:
            # Convert to lowercase and remove 'scalar_' prefix if present
            standard_name = prop.lower().replace('scalar_', '')
            property_map[standard_name] = prop
        
        # Handle pred property
        if 'pred' in property_map:
            preds = vertex[property_map['pred']]
            print(f"Found pred property as '{property_map['pred']}'")
        else:
            print("Warning: No pred property found, using default value 0")
            preds = np.zeros(len(x))
            
        # Handle pwood property
        if 'pwood' in property_map:
            pwood = vertex[property_map['pwood']]
            print(f"Found pwood property as '{property_map['pwood']}'")
        else:
            print("Warning: No pwood property found, using default value 0.5")
            pwood = np.ones(len(x)) * 0.5
        
        # Create initial DataFrame
        df = pd.DataFrame({
            'x': x, 
            'y': y, 
            'z': z, 
            'pred': preds, 
            'pwood': pwood
        })
        
        # Look for reflectance property (could have various names)
        reflectance_col = None
        reflectance_candidates = ['reflectance', 'intensity', 'reflect']
        
        for candidate in reflectance_candidates:
            if candidate in property_map:
                reflectance_col = property_map[candidate]
                df['reflectance'] = vertex[reflectance_col]
                print(f"Found reflectance data in column '{reflectance_col}'")
                
                # Normalize reflectance to 0-1 range if needed
                if df['reflectance'].max() > 1.0:
                    min_val = df['reflectance'].min()
                    max_val = df['reflectance'].max()
                    df['reflectance'] = (df['reflectance'] - min_val) / (max_val - min_val)
                    print(f"Normalized reflectance from range [{min_val}, {max_val}] to [0, 1]")
                break
        
        if not reflectance_col:
            print("No reflectance property found. Looking for any other scalar properties...")
            
            # Look for any other scalar properties that might be useful
            for prop in properties:
                standard_name = prop.lower().replace('scalar_', '')
                if standard_name not in ['x', 'y', 'z', 'pred', 'pwood']:
                    try:
                        df['reflectance'] = vertex[prop]
                        print(f"Using '{prop}' as reflectance substitute")
                        
                        # Normalize to 0-1 range
                        min_val = df['reflectance'].min()
                        max_val = df['reflectance'].max()
                        if min_val != max_val:
                            df['reflectance'] = (df['reflectance'] - min_val) / (max_val - min_val)
                            print(f"Normalized '{prop}' from range [{min_val}, {max_val}] to [0, 1]")
                        reflectance_col = prop
                        break
                    except:
                        continue
        
        if not reflectance_col:
            print("No suitable reflectance substitute found. Creating synthetic reflectance...")
            # Create synthetic reflectance based on position
            df['reflectance'] = (df['x'] % 1 + df['y'] % 1 + df['z'] % 1) / 3
        
        print(f"Successfully loaded {len(df)} points")
        print(f"Data columns: {df.columns.tolist()}")
        print(f"Sample data:\n{df.head()}")
        
        # Print pred distribution
        pred_counts = df['pred'].value_counts()
        print(f"pred distribution:\n{pred_counts}")
        
        # Print reflectance statistics
        print(f"Reflectance statistics: min={df['reflectance'].min()}, max={df['reflectance'].max()}, mean={df['reflectance'].mean():.4f}")
        
        return df
        
    except Exception as e:
        print(f"Error loading PLY file: {e}")
        raise

def apply_smoothing(image, smoothing_level=1):
    """Apply smoothing to an image"""
    if smoothing_level <= 0:
        return image
    
    # Convert to PIL image if it's a datashader image
    if not isinstance(image, Image.Image):
        pil_img = image.to_pil()
    else:
        pil_img = image
    
    # Apply Gaussian blur
    for _ in range(smoothing_level):
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.8))
    
    return pil_img

def render_predictions_xray_view(df, output_path, width=6000, height=6000, smoothing_level=1):
    """Render tree with X-ray effect and red wood based on pwood"""
    print("Creating smoothed predictions X-ray visualization...")
    
    # Calculate data ranges to determine aspect ratio
    x_range = (df['x'].min(), df['x'].max())
    y_range = (df['y'].min(), df['y'].max())
    z_range = (df['z'].min(), df['z'].max())
    
    x_span = x_range[1] - x_range[0]
    y_span = y_range[1] - y_range[0]
    z_span = z_range[1] - z_range[0]
    
    # Determine which side view is more informative
    xz_area = x_span * z_span
    yz_area = y_span * z_span
    use_xz = xz_area >= yz_area
    
    print(f"Side view extents: XZ area = {xz_area:.2f}, YZ area = {yz_area:.2f}")
    print(f"Using {'XZ' if use_xz else 'YZ'} projection for side view")
    
    # Create separate dataframes for wood and leaves
    wood_df = df[df['pred'] == 1].copy()
    leaf_df = df[df['pred'] == 0].copy()
    
    print(f"Wood points: {len(wood_df)}, Leaf points: {len(leaf_df)}")
    
    # Normalize pwood for coloring
    if 'pwood' in wood_df.columns:
        min_val = wood_df['pwood'].min()
        max_val = wood_df['pwood'].max()
        if min_val != max_val:
            wood_df['pwood_norm'] = (wood_df['pwood'] - min_val) / (max_val - min_val)
        else:
            wood_df['pwood_norm'] = 0.5 * np.ones(len(wood_df))  # Use middle value if all pwood values are the same
    else:
        # Create synthetic values if pwood doesn't exist
        wood_df['pwood_norm'] = np.random.random(len(wood_df))
    
    # Side view (XZ or YZ)
    print("\nCreating predictions side view...")
    
    # Create canvas with proper aspect ratio for side view
    aspect_ratio = (x_span if use_xz else y_span) / z_span
    
    # Adjust canvas dimensions to maintain aspect ratio
    if aspect_ratio > 1:
        # Wider than tall
        canvas_width = width
        canvas_height = int(width / aspect_ratio)
    else:
        # Taller than wide
        canvas_height = height
        canvas_width = int(height * aspect_ratio)
    
    # Increase resolution for better detail and smoother rendering
    canvas_width = int(canvas_width * 1.5)
    canvas_height = int(canvas_height * 1.5)
    
    print(f"Canvas dimensions: {canvas_width}x{canvas_height}")
    
    # Create canvas with proper dimensions
    canvas = ds.Canvas(plot_width=canvas_width, plot_height=canvas_height)
    
    # Render leaves with density-based alpha transparency
    print("Rendering leaves with density-based alpha (more transparent in dense areas)...")
    if len(leaf_df) > 0:
        # Aggregate density for leaf presence
        leaf_agg = canvas.points(
            leaf_df, 
            'x' if use_xz else 'y',  # x-axis or y-axis depending on chosen view
            'z',  # z-axis (height)
            ds.count()  # Just count points to determine presence
        )
        
        # Use density-based alpha: high density -> more transparent, low density -> more opaque
        leaf_img = tf.shade(leaf_agg, cmap='black', how='linear', alpha=200, min_alpha=40)
        
        # Apply a larger spread to fill gaps between points
        leaf_img = tf.spread(leaf_img, px=3)
        
        # Convert to PIL image
        leaf_pil = tf.Image(leaf_img).to_pil()
    else:
        print("No leaf points found")
        leaf_pil = None
    
    # Render wood with red colormap based on pwood
    print("Rendering wood with red colormap based on pwood...")
    if len(wood_df) > 0:
        # Use pwood_norm for coloring
        wood_agg = canvas.points(
            wood_df, 
            'x' if use_xz else 'y',  # x-axis or y-axis depending on chosen view
            'z',  # z-axis (height)
            ds.mean('pwood_norm')  # Use normalized pwood for coloring
        )
        
        # Create a red colormap with varying intensity based on pwood
        # Darker red for lower pwood, brighter red for higher pwood
        red_colors = [
            (0.5, 0.0, 0.0, 1.0),  # Dark red
            (0.6, 0.0, 0.0, 1.0),
            (0.7, 0.0, 0.0, 1.0),
            (0.8, 0.0, 0.0, 1.0),
            (0.9, 0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0, 1.0),  # Bright red
        ]
        red_cmap = LinearSegmentedColormap.from_list('red_pwood', red_colors)
        
        # Apply the colormap
        wood_img = tf.shade(wood_agg, cmap=red_cmap, how='linear')
        
        # Apply a larger spread to fill gaps between points
        wood_img = tf.spread(wood_img, px=3)
        
        # Convert to PIL image
        wood_pil = tf.Image(wood_img).to_pil()
        
        # Apply a slight blur to smooth out the image
        wood_pil = wood_pil.filter(ImageFilter.GaussianBlur(radius=1))
        
        # Then apply sharpening to enhance details while keeping the smoothness
        wood_pil = wood_pil.filter(ImageFilter.SHARPEN)
    else:
        print("No wood points found")
        wood_pil = None
    
    # Composite images for side view
    print("Compositing predictions side view...")
    
    # Create a white background image
    background = Image.new('RGBA', (canvas_width, canvas_height), color='white')
    
    # Composite images - leaves first, then wood on top
    side_view = background
    
    if leaf_pil is not None:
        side_view = Image.alpha_composite(side_view.convert('RGBA'), leaf_pil)
    
    if wood_pil is not None:
        side_view = Image.alpha_composite(side_view.convert('RGBA'), wood_pil)
    
    # Resize to original dimensions if needed
    if canvas_width > width or canvas_height > height:
        # Calculate new dimensions while preserving aspect ratio
        if canvas_width / canvas_height > width / height:
            new_width = width
            new_height = int(canvas_height * (width / canvas_width))
        else:
            new_height = height
            new_width = int(canvas_width * (height / canvas_height))
        
        # Resize with high quality
        side_view = side_view.resize((new_width, new_height), Image.LANCZOS)
    
    # Save the most informative side view
    side_view_path = output_path.replace('.png', '_side.png')
    side_view.save(side_view_path, quality=100)  # Maximum quality for PNG
    print(f"Saved predictions side view to {side_view_path} (using {'XZ' if use_xz else 'YZ'} projection)")
    
    # Bottom view (XY)
    print("\nCreating predictions bottom view (XY)...")
    
    # Create canvas with proper aspect ratio for XY view
    aspect_ratio = x_span / y_span
    
    # Adjust canvas dimensions to maintain aspect ratio
    if aspect_ratio > 1:
        # Wider than tall
        bottom_width = width
        bottom_height = int(width / aspect_ratio)
    else:
        # Taller than wide
        bottom_height = height
        bottom_width = int(height * aspect_ratio)
    
    # Increase resolution for better detail and smoother rendering
    bottom_width = int(bottom_width * 1.5)
    bottom_height = int(bottom_height * 1.5)
    
    print(f"Bottom view dimensions: {bottom_width}x{bottom_height}")
    
    # Create canvas with proper dimensions for bottom view
    bottom_canvas = ds.Canvas(plot_width=bottom_width, plot_height=bottom_height)
    
    # Render leaves for bottom view with density-based alpha transparency
    print("Rendering leaves for bottom view with density-based alpha...")
    if len(leaf_df) > 0:
        # Aggregate density for leaf presence
        bottom_leaf_agg = bottom_canvas.points(
            leaf_df, 
            'x',  # x-axis 
            'y',  # y-axis
            ds.count()  # Just count points to determine presence
        )
        
        # Use same density-based alpha approach
        bottom_leaf_img = tf.shade(bottom_leaf_agg, cmap='black', how='linear', alpha=200, min_alpha=40)
        
        # Apply a larger spread to fill gaps between points
        bottom_leaf_img = tf.spread(bottom_leaf_img, px=3)
        
        # Convert to PIL image
        bottom_leaf_pil = tf.Image(bottom_leaf_img).to_pil()
    else:
        print("No leaf points found")
        bottom_leaf_pil = None
    
    # Render wood for bottom view with red colormap
    print("Rendering wood for bottom view with red colormap...")
    if len(wood_df) > 0:
        # Use pwood_norm for coloring
        bottom_wood_agg = bottom_canvas.points(
            wood_df, 
            'x',  # x-axis
            'y',  # y-axis
            ds.mean('pwood_norm')  # Use normalized pwood for coloring
        )
        
        # Apply the same colormap
        bottom_wood_img = tf.shade(bottom_wood_agg, cmap=red_cmap, how='linear')
        
        # Apply a larger spread to fill gaps between points
        bottom_wood_img = tf.spread(bottom_wood_img, px=3)
        
        # Convert to PIL image
        bottom_wood_pil = tf.Image(bottom_wood_img).to_pil()
        
        # Apply a slight blur to smooth out the image
        bottom_wood_pil = bottom_wood_pil.filter(ImageFilter.GaussianBlur(radius=1))
        
        # Then apply sharpening to enhance details while keeping the smoothness
        bottom_wood_pil = bottom_wood_pil.filter(ImageFilter.SHARPEN)
    else:
        print("No wood points found")
        bottom_wood_pil = None
    
    # Composite images for bottom view
    print("Compositing predictions bottom view...")
    
    # Create a white background image
    bottom_background = Image.new('RGBA', (bottom_width, bottom_height), color='white')
    
    # Composite images - leaves first, then wood on top
    bottom_view = bottom_background
    
    if bottom_leaf_pil is not None:
        bottom_view = Image.alpha_composite(bottom_view.convert('RGBA'), bottom_leaf_pil)
    
    if bottom_wood_pil is not None:
        bottom_view = Image.alpha_composite(bottom_view.convert('RGBA'), bottom_wood_pil)
    
    # Resize to original dimensions if needed
    if bottom_width > width or bottom_height > height:
        # Calculate new dimensions while preserving aspect ratio
        if bottom_width / bottom_height > width / height:
            new_width = width
            new_height = int(bottom_height * (width / bottom_width))
        else:
            new_height = height
            new_width = int(bottom_width * (height / bottom_height))
        
        # Resize with high quality
        bottom_view = bottom_view.resize((new_width, new_height), Image.LANCZOS)
    
    # Save the bottom view
    bottom_view_path = output_path.replace('.png', '_bottom.png')
    bottom_view.save(bottom_view_path, quality=100)  # Maximum quality for PNG
    print(f"Saved predictions bottom view to {bottom_view_path}")
    
    return side_view, bottom_view

def render_pwood_view(df, output_path, width=6000, height=6000, colormap='inferno'):
    """Render tree with X-ray effect and specified colormap for wood based on raw pwood values
    
    Args:
        df: DataFrame containing point cloud data
        output_path: Path to save the output images
        width: Width of the output image
        height: Height of the output image
        colormap: Name of matplotlib colormap to use for wood points (e.g., 'inferno', 'turbo', 'viridis')
    """
    print(f"Creating pwood-colored X-ray views using {colormap} colormap...")
    
    # Calculate data ranges to determine aspect ratio
    x_range = (df['x'].min(), df['x'].max())
    y_range = (df['y'].min(), df['y'].max())
    z_range = (df['z'].min(), df['z'].max())
    
    x_span = x_range[1] - x_range[0]
    y_span = y_range[1] - y_range[0]
    z_span = z_range[1] - z_range[0]
    
    # Determine which side view is more informative
    xz_area = x_span * z_span
    yz_area = y_span * z_span
    use_xz = xz_area >= yz_area
    
    print(f"Side view extents: XZ area = {xz_area:.2f}, YZ area = {yz_area:.2f}")
    print(f"Using {'XZ' if use_xz else 'YZ'} projection for side view")
    
    # Create separate dataframes for wood and leaves
    wood_df = df[df['pred'] == 1].copy()
    leaf_df = df[df['pred'] == 0].copy()
    
    print(f"Wood points: {len(wood_df)}, Leaf points: {len(leaf_df)}")
    
    # Normalize pwood values to full 0-1 range for better colormap utilization
    if len(wood_df) > 0 and 'pwood' in wood_df.columns:
        pwood_min = wood_df['pwood'].min()
        pwood_max = wood_df['pwood'].max()
        pwood_range = pwood_max - pwood_min
        
        print(f"Original pwood range: {pwood_min:.3f} to {pwood_max:.3f} (span: {pwood_range:.3f})")
        
        if pwood_range > 0:
            # Normalize to 0-1 range
            wood_df['pwood_normalized'] = (wood_df['pwood'] - pwood_min) / pwood_range
            print(f"Normalized pwood to 0-1 range for full colormap utilization")
        else:
            # All values are the same
            wood_df['pwood_normalized'] = np.ones(len(wood_df)) * 0.5
            print(f"All pwood values are identical ({pwood_min:.3f}), using 0.5 for visualization")
    else:
        # Fallback if no pwood data
        wood_df['pwood_normalized'] = np.ones(len(wood_df)) * 0.5 if len(wood_df) > 0 else []
        print("No pwood data found, using 0.5 for visualization")
    
    # Side view setup
    print("\nCreating pwood side view...")
    
    # Create canvas with proper aspect ratio for side view
    aspect_ratio = (x_span if use_xz else y_span) / z_span
    
    # Adjust canvas dimensions to maintain aspect ratio
    if aspect_ratio > 1:
        canvas_width = width
        canvas_height = int(width / aspect_ratio)
    else:
        canvas_height = height
        canvas_width = int(height * aspect_ratio)
    
    # Increase resolution for better detail
    canvas_width = int(canvas_width * 1.5)
    canvas_height = int(canvas_height * 1.5)
    
    print(f"Canvas dimensions: {canvas_width}x{canvas_height}")
    
    # Create canvas
    canvas = ds.Canvas(plot_width=canvas_width, plot_height=canvas_height)
    
    # Render leaves with density-based alpha transparency
    print("Rendering leaves with density-based alpha (more transparent in dense areas)...")
    if len(leaf_df) > 0:
        # Aggregate density for leaf presence
        leaf_agg = canvas.points(
            leaf_df, 
            'x' if use_xz else 'y',
            'z',
            ds.count()
        )
        
        # Use density-based alpha: high density -> more transparent, low density -> more opaque
        leaf_img = tf.shade(leaf_agg, cmap='black', how='linear', alpha=200, min_alpha=40)
        
        # Apply a larger spread to fill gaps between points
        leaf_img = tf.spread(leaf_img, px=3)
        
        # Convert to PIL image
        leaf_pil = tf.Image(leaf_img).to_pil()
    else:
        print("No leaf points found")
        leaf_pil = None

    # Render wood with specified colormap (unchanged)
    print(f"Rendering wood with {colormap} colormap...")
    if len(wood_df) > 0:
        wood_agg = canvas.points(
            wood_df, 
            'x' if use_xz else 'y',
            'z',
            ds.mean('pwood_normalized')  # Use normalized pwood values
        )
        
        # Use specified matplotlib colormap
        from matplotlib import colormaps
        cmap = colormaps[colormap]
        wood_cmap = LinearSegmentedColormap.from_list(colormap, cmap(np.linspace(0, 1, 256)))
        
        # Use explicit span to force full colormap range utilization
        wood_img = tf.shade(wood_agg, cmap=wood_cmap, how='linear', span=(0, 1))
        wood_img = tf.spread(wood_img, px=3)
        wood_pil = tf.Image(wood_img).to_pil()
        
        # Apply slight blur and sharpen for better visualization
        wood_pil = wood_pil.filter(ImageFilter.GaussianBlur(radius=1))
        wood_pil = wood_pil.filter(ImageFilter.SHARPEN)
    else:
        print("No wood points found")
        wood_pil = None
    
    # Composite side view
    print("Compositing pwood side view...")
    background = Image.new('RGBA', (canvas_width, canvas_height), color='white')
    side_view = background
    
    if leaf_pil is not None:
        side_view = Image.alpha_composite(side_view.convert('RGBA'), leaf_pil)
    if wood_pil is not None:
        side_view = Image.alpha_composite(side_view.convert('RGBA'), wood_pil)
    
    # Resize if needed
    if canvas_width > width or canvas_height > height:
        if canvas_width / canvas_height > width / height:
            new_width = width
            new_height = int(canvas_height * (width / canvas_width))
        else:
            new_height = height
            new_width = int(canvas_width * (height / canvas_height))
        side_view = side_view.resize((new_width, new_height), Image.LANCZOS)
    
    # Save side view
    side_view_path = output_path.replace('.png', '_side.png')
    side_view.save(side_view_path, quality=100)
    print(f"Saved pwood side view to {side_view_path} (using {'XZ' if use_xz else 'YZ'} projection)")
    
    # Bottom view (XY)
    print("\nCreating pwood bottom view (XY)...")
    
    # Setup bottom view canvas
    aspect_ratio = x_span / y_span
    if aspect_ratio > 1:
        bottom_width = width
        bottom_height = int(width / aspect_ratio)
    else:
        bottom_height = height
        bottom_width = int(height * aspect_ratio)
    
    bottom_width = int(bottom_width * 1.5)
    bottom_height = int(bottom_height * 1.5)
    
    print(f"Bottom view dimensions: {bottom_width}x{bottom_height}")
    bottom_canvas = ds.Canvas(plot_width=bottom_width, plot_height=bottom_height)
    
    # Render leaves for bottom view
    print("Rendering leaves for bottom view...")
    if len(leaf_df) > 0:
        bottom_leaf_agg = bottom_canvas.points(leaf_df, 'x', 'y', ds.count())
        # Use same corrected alpha approach: high density -> min_alpha (transparent), low density -> alpha (opaque)
        bottom_leaf_img = tf.shade(bottom_leaf_agg, cmap='black', how='linear', alpha=200, min_alpha=40)
        bottom_leaf_img = tf.spread(bottom_leaf_img, px=3)
        bottom_leaf_pil = tf.Image(bottom_leaf_img).to_pil()
    else:
        bottom_leaf_pil = None
    
    # Render wood for bottom view
    print("Rendering wood for bottom view...")
    if len(wood_df) > 0:
        bottom_wood_agg = bottom_canvas.points(wood_df, 'x', 'y', ds.mean('pwood_normalized'))
        bottom_wood_img = tf.shade(bottom_wood_agg, cmap=wood_cmap, how='linear', span=(0, 1))
        bottom_wood_img = tf.spread(bottom_wood_img, px=3)
        bottom_wood_pil = tf.Image(bottom_wood_img).to_pil()
        # Apply slight blur and sharpen for better visualization
        bottom_wood_pil = bottom_wood_pil.filter(ImageFilter.GaussianBlur(radius=1))
        bottom_wood_pil = bottom_wood_pil.filter(ImageFilter.SHARPEN)
    else:
        bottom_wood_pil = None
    
    # Composite bottom view
    print("Compositing pwood bottom view...")
    bottom_background = Image.new('RGBA', (bottom_width, bottom_height), color='white')
    bottom_view = bottom_background
    
    if bottom_leaf_pil is not None:
        bottom_view = Image.alpha_composite(bottom_view.convert('RGBA'), bottom_leaf_pil)
    if bottom_wood_pil is not None:
        bottom_view = Image.alpha_composite(bottom_view.convert('RGBA'), bottom_wood_pil)
    
    # Resize if needed
    if bottom_width > width or bottom_height > height:
        if bottom_width / bottom_height > width / height:
            new_width = width
            new_height = int(bottom_height * (width / bottom_width))
        else:
            new_height = height
            new_width = int(bottom_width * (height / bottom_height))
        bottom_view = bottom_view.resize((new_width, new_height), Image.LANCZOS)
    
    # Save bottom view
    bottom_view_path = output_path.replace('.png', '_bottom.png')
    bottom_view.save(bottom_view_path, quality=100)
    print(f"Saved pwood bottom view to {bottom_view_path}")
    
    return side_view, bottom_view

def main():
    # Check if a file path was provided as a command-line argument
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        # Check if colormap is provided as second argument
        colormap = sys.argv[2] if len(sys.argv) > 2 else 'inferno'
    else:
        # Default file path if none provided
        input_file = input("Enter path to PLY file: ")
        colormap = 'inferno'  # Default colormap
    
    # Load the PLY file
    print(f"\nLoading PLY file: {input_file}")
    df = load_ply(input_file)
    
    # Determine output paths
    output_dir = os.path.dirname(input_file) if os.path.dirname(input_file) else os.path.dirname(os.path.abspath(__file__))
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # Create output paths
    predictions_output = os.path.join(output_dir, f"{base_name}_predictions.png")
    pwood_output = os.path.join(output_dir, f"{base_name}_pwood.png")
    
    # Render views
    render_predictions_xray_view(df, predictions_output)
    render_pwood_view(df, pwood_output, colormap=colormap)

if __name__ == "__main__":
    main()