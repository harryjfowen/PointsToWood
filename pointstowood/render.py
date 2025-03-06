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
    """Load PLY file with xyz coordinates, label, pwood, and reflectance columns using plyfile"""
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
        
        # Handle label property
        if 'label' in property_map:
            labels = vertex[property_map['label']]
            print(f"Found label property as '{property_map['label']}'")
        else:
            print("Warning: No label property found, using default value 0")
            labels = np.zeros(len(x))
            
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
            'label': labels, 
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
                if standard_name not in ['x', 'y', 'z', 'label', 'pwood']:
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
        
        # Print label distribution
        label_counts = df['label'].value_counts()
        print(f"Label distribution:\n{label_counts}")
        
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

def render_textured_xray_view(df, output_path, width=6000, height=6000, smoothing_level=1):
    """Render tree with X-ray effect and red wood based on pwood"""
    print("Creating smoothed textured X-ray visualization...")
    
    # Calculate data ranges to determine aspect ratio
    x_range = (df['x'].min(), df['x'].max())
    y_range = (df['y'].min(), df['y'].max())
    z_range = (df['z'].min(), df['z'].max())
    
    x_span = x_range[1] - x_range[0]
    y_span = y_range[1] - y_range[0]
    z_span = z_range[1] - z_range[0]
    
    # Create separate dataframes for wood and leaves
    wood_df = df[df['label'] == 1].copy()
    leaf_df = df[df['label'] == 0].copy()
    
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
    
    # Side view (XZ)
    print("\nCreating textured side view (XZ)...")
    
    # Create canvas with proper aspect ratio for XZ view
    aspect_ratio = x_span / z_span
    
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
    
    # Render leaves with solid black color
    print("Rendering leaves with solid black color...")
    if len(leaf_df) > 0:
        # Aggregate density for leaf presence
        leaf_agg = canvas.points(
            leaf_df, 
            'x',  # x-axis 
            'z',  # z-axis (height)
            ds.count()  # Just count points to determine presence
        )
        
        # Create a solid black colormap with fixed opacity
        black_cmap = LinearSegmentedColormap.from_list('black', [(0, 0, 0, 0.7), (0, 0, 0, 0.7)])
        
        # Apply the colormap
        leaf_img = tf.shade(leaf_agg, cmap=black_cmap, how='linear')
        
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
            'x',  # x-axis
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
    print("Compositing textured side view...")
    
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
    
    # Save the side view
    side_view_path = output_path.replace('.png', '_side.png')
    side_view.save(side_view_path, quality=100)  # Maximum quality for PNG
    print(f"Saved textured side view to {side_view_path}")
    
    # Bottom view (XY)
    print("\nCreating textured bottom view (XY)...")
    
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
    
    # Render leaves for bottom view with solid black color
    print("Rendering leaves for bottom view with solid black color...")
    if len(leaf_df) > 0:
        # Aggregate density for leaf presence
        bottom_leaf_agg = bottom_canvas.points(
            leaf_df, 
            'x',  # x-axis 
            'y',  # y-axis
            ds.count()  # Just count points to determine presence
        )
        
        # Apply the same black colormap
        bottom_leaf_img = tf.shade(bottom_leaf_agg, cmap=black_cmap, how='linear')
        
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
    print("Compositing textured bottom view...")
    
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
    print(f"Saved textured bottom view to {bottom_view_path}")
    
    return side_view, bottom_view

def render_vik_xray_view(df, output_path, width=6000, height=6000):
    """Render tree with X-ray effect and vik colormap for wood based on pwood"""
    print("Creating vik-colored X-ray views...")
    
    # Calculate data ranges to determine aspect ratio
    x_range = (df['x'].min(), df['x'].max())
    y_range = (df['y'].min(), df['y'].max())
    z_range = (df['z'].min(), df['z'].max())
    
    x_span = x_range[1] - x_range[0]
    y_span = y_range[1] - y_range[0]
    z_span = z_range[1] - z_range[0]
    
    # Create separate dataframes for wood and leaves
    wood_df = df[df['label'] == 1].copy()
    leaf_df = df[df['label'] == 0].copy()
    
    print(f"Wood points: {len(wood_df)}, Leaf points: {len(leaf_df)}")
    
    # Side view (XZ)
    print("\nCreating vik side view (XZ)...")
    
    # Create canvas with proper aspect ratio for XZ view
    aspect_ratio = x_span / z_span
    
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
    
    # Render leaves with solid black color
    print("Rendering leaves with solid black color...")
    if len(leaf_df) > 0:
        # Aggregate density for leaf presence
        leaf_agg = canvas.points(
            leaf_df, 
            'x',  # x-axis 
            'z',  # z-axis (height)
            ds.count()  # Just count points to determine presence
        )
        
        # Create a solid black colormap with fixed opacity
        black_cmap = LinearSegmentedColormap.from_list('black', [(0, 0, 0, 0.7), (0, 0, 0, 0.7)])
        
        # Apply the colormap
        leaf_img = tf.shade(leaf_agg, cmap=black_cmap, how='linear')
        
        # Apply a larger spread to fill gaps between points
        leaf_img = tf.spread(leaf_img, px=3)
        
        # Convert to PIL image
        leaf_pil = tf.Image(leaf_img).to_pil()
    else:
        print("No leaf points found")
        leaf_pil = None
    
    # Render wood with vik colormap based on pwood
    print("Rendering wood with vik colormap based on pwood...")
    if len(wood_df) > 0:
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
        
        # Use pwood_norm for coloring
        wood_agg = canvas.points(
            wood_df, 
            'x',  # x-axis
            'z',  # z-axis (height)
            ds.mean('pwood_norm')  # Use normalized pwood for coloring
        )
        
        # Get the vik colormap from scicomap
        # sc_map = sc.ScicoDiverging(cmap='pride')
        # vik_cmap = sc_map.get_mpl_color_map()

        sc_map = sc.ScicoMiscellaneous(cmap='rainbow-kov')
        vik_cmap = sc_map.get_mpl_color_map()
        
        # Make sure the colormap has full opacity
        vik_cmap._rgba_bad = (0, 0, 0, 0)  # Transparent for NaN values
        
        # Apply the colormap
        wood_img = tf.shade(wood_agg, cmap=vik_cmap, how='linear')
        
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
    print("Compositing vik side view...")
    
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
    
    # Save the side view
    side_view_path = output_path.replace('.png', '_side.png')
    side_view.save(side_view_path, quality=100)  # Maximum quality for PNG
    print(f"Saved vik side view to {side_view_path}")
    
    # Bottom view (XY)
    print("\nCreating vik bottom view (XY)...")
    
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
    
    # Render leaves for bottom view with solid black color
    print("Rendering leaves for bottom view with solid black color...")
    if len(leaf_df) > 0:
        # Aggregate density for leaf presence
        bottom_leaf_agg = bottom_canvas.points(
            leaf_df, 
            'x',  # x-axis 
            'y',  # y-axis
            ds.count()  # Just count points to determine presence
        )
        
        # Apply the same black colormap
        bottom_leaf_img = tf.shade(bottom_leaf_agg, cmap=black_cmap, how='linear')
        
        # Apply a larger spread to fill gaps between points
        bottom_leaf_img = tf.spread(bottom_leaf_img, px=3)
        
        # Convert to PIL image
        bottom_leaf_pil = tf.Image(bottom_leaf_img).to_pil()
    else:
        print("No leaf points found")
        bottom_leaf_pil = None
    
    # Render wood for bottom view with vik colormap
    print("Rendering wood for bottom view with vik colormap...")
    if len(wood_df) > 0:
        # Use pwood_norm for coloring
        bottom_wood_agg = bottom_canvas.points(
            wood_df, 
            'x',  # x-axis
            'y',  # y-axis
            ds.mean('pwood_norm')  # Use normalized pwood for coloring
        )
        
        # Apply the same colormap
        bottom_wood_img = tf.shade(bottom_wood_agg, cmap=vik_cmap, how='linear')
        
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
    print("Compositing vik bottom view...")
    
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
    print(f"Saved vik bottom view to {bottom_view_path}")
    
    return side_view, bottom_view

def main():
    # Check if a file path was provided as a command-line argument
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        # Default file path if none provided
        input_file = input("Enter path to PLY file: ")
    
    # Load the PLY file
    print(f"\nLoading PLY file: {input_file}")
    df = load_ply(input_file)
    
    # Determine output paths
    output_dir = os.path.dirname(input_file) if os.path.dirname(input_file) else os.path.dirname(os.path.abspath(__file__))
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # Create the red wood visualization with smoothing
    print(f"\nCreating smoothed textured X-ray visualization (red wood)...")
    red_output_path = os.path.join(output_dir, f"{base_name}_textured_xray.png")
    render_textured_xray_view(df, red_output_path, width=6000, height=6000, smoothing_level=1)
    
    # Create the vik wood visualization with smoothing
    print(f"\nCreating smoothed vik-colored X-ray visualization...")
    vik_output_path = os.path.join(output_dir, f"{base_name}_vik_xray.png")
    render_vik_xray_view(df, vik_output_path, width=6000, height=6000)
    
    print("\nVisualizations complete!")
    print(f"1. Smoothed red wood X-ray: {red_output_path}")
    print(f"2. Smoothed vik wood X-ray: {vik_output_path}")

if __name__ == "__main__":
    main()