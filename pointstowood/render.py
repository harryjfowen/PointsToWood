import numpy as np
import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf
from plyfile import PlyData
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageFilter
import sys
import os
import colorcet as cc
from colorcet import fire

def load_ply(filepath):
    """Load PLY file with xyz coordinates, prediction, and pwood columns"""
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
        
        # Create case-insensitive mapping
        property_map = {}
        for prop in properties:
            standard_name = prop.lower().replace('scalar_', '')
            property_map[standard_name] = prop
        
        # Handle prediction property
        if 'prediction' in property_map:
            preds = vertex[property_map['prediction']]
            print(f"Found prediction property as '{property_map['prediction']}'")
        else:
            print("Warning: No prediction property found, using default value 0")
            preds = np.zeros(len(x))
            
        # Handle pwood property
        if 'pwood' in property_map:
            pwood = vertex[property_map['pwood']]
            print(f"Found pwood property as '{property_map['pwood']}'")
        else:
            print("Warning: No pwood property found, using default value 0.5")
            pwood = np.ones(len(x)) * 0.5
        
        # Handle path_length property
        if 'path_length' in property_map:
            path_length = vertex[property_map['path_length']]
            print(f"Found path_length property as '{property_map['path_length']}'")
        else:
            print("No path_length property found")
            path_length = None
        
        # Create DataFrame
        df = pd.DataFrame({
            'x': x, 'y': y, 'z': z, 'prediction': preds, 'pwood': pwood
        })
        
        # Add path_length if available
        if path_length is not None:
            df['path_length'] = path_length
        
        print(f"Successfully loaded {len(df)} points")
        print(f"prediction distribution:\n{df['prediction'].value_counts()}")
        
        return df
        
    except Exception as e:
        print(f"Error loading PLY file: {e}")
        raise

def create_canvas(width, height, aspect_ratio):
    """Create canvas with proper aspect ratio and increased resolution"""
    if aspect_ratio > 1:
        canvas_width = width
        canvas_height = int(width / aspect_ratio)
    else:
        canvas_height = height
        canvas_width = int(height * aspect_ratio)
    
    # Increase resolution for better detail
    canvas_width = int(canvas_width * 1.5)
    canvas_height = int(canvas_height * 1.5)
    
    return ds.Canvas(plot_width=canvas_width, plot_height=canvas_height), canvas_width, canvas_height

def create_colormap(colormap_name):
    """Create custom colormap for wood points"""
    print(f"Creating colormap: '{colormap_name}'")
    
    if colormap_name.lower() == 'fire':
        print("Using custom fire colormap (inverted and clipped)")
        # Use inverted and cropped fire colormap from colorcet
        custom_colors = [(0.0, 0.0, 0.0, 1.0)]  # Black for leaves
        end_idx = int(256 * 0.75)
        for i in range(256):
            palette_idx = int(i * end_idx / 255)
            color = fire[255 - palette_idx]  # Invert the index
            custom_colors.append(color)
        return LinearSegmentedColormap.from_list('custom_fire', custom_colors)
    else:
        print(f"Using matplotlib colormap: '{colormap_name}'")
        # Use matplotlib colormap
        from matplotlib import colormaps
        try:
            base_cmap = colormaps[colormap_name]
            custom_colors = [(0.0, 0.0, 0.0, 1.0)]  # Black for leaves
            for i in range(256):
                color = base_cmap(i / 255.0)
                custom_colors.append(color)
            return LinearSegmentedColormap.from_list('custom_colors', custom_colors)
        except KeyError:
            print(f"Warning: Colormap '{colormap_name}' not found, using 'fire' as fallback")
            return create_colormap('fire')

def create_fire_colormap():
    """Create inverted and cropped fire colormap for wood points"""
    # For datashader, we need to create a custom palette that includes black for leaves
    # and the inverted fire colors for wood
    custom_palette = ['black']  # Black for leaves (prediction == 0)
    
    # Add inverted fire colormap colors for wood (prediction == 1)
    # Invert so red = high pwood, yellow = low pwood
    # Skip last 25% of original palette (which becomes first 25% after inversion)
    end_idx = int(len(fire) * 0.75)  # End at 75% of the original palette
    for i in range(len(fire)):
        # Map i to the cropped range and invert
        palette_idx = int(i * end_idx / len(fire))
        if palette_idx < len(fire):
            color = fire[len(fire) - 1 - palette_idx]  # Invert the index
            custom_palette.append(color)
    
    return custom_palette

def apply_selective_alpha(final_img, alpha_channel, df=None):
    """Apply selective alpha - strong alpha for leaves, full opacity for wood points"""
    final_data = final_img.getdata()
    alpha_data = alpha_channel.getdata()
    
    new_data = []
    for i, (r, g, b, a) in enumerate(final_data):
        if r < 50 and g < 50 and b < 50:  # Mostly black (leaves)
            # Apply full density-based alpha for leaves
            new_alpha = alpha_data[i]
        else:  # Colored (wood)
            # Keep wood points fully opaque - no effects, just clean pwood coloring
            new_alpha = 255  # Full opacity for wood points
        
        new_data.append((r, g, b, new_alpha))
    
    final_img.putdata(new_data)
    return final_img

def render_view(df, canvas, x_col, y_col, agg_method, colormap, span=None, is_predictions=False):
    """Render a single view with X-ray effect and wood smoothing"""
    # Render all points together
    all_agg = canvas.points(df, x_col, y_col, agg_method)
    
    # Apply colormap
    if span:
        all_img = tf.shade(all_agg, cmap=colormap, how='linear', span=span)
    else:
        all_img = tf.shade(all_agg, cmap=colormap, how='linear')
    
    all_img = tf.spread(all_img, px=2)  # Reduced from 3 to 2
    all_pil = tf.Image(all_img).to_pil()
    
    # Apply very subtle smoothing to RGB channels only (before alpha)
    from PIL import ImageFilter
    # Apply minimal Gaussian blur for smooth wood appearance
    all_pil = all_pil.filter(ImageFilter.GaussianBlur(radius=0.5))  # Reduced from 0.8 to 0.5
    
    # Create density-based X-ray effect
    leaf_df = df[df['prediction'] == 0]
    leaf_density_agg = canvas.points(leaf_df, x_col, y_col, ds.count())
    
    # Create alpha mask with strong transparency effect
    alpha_img = tf.shade(leaf_density_agg, cmap='black', how='linear', alpha=255, min_alpha=10)
    alpha_img = tf.spread(alpha_img, px=2)  # Reduced from 3 to 2
    alpha_pil = tf.Image(alpha_img).to_pil()
    
    # Apply X-ray effect - preserve original alpha to prevent wood fading
    all_rgba = all_pil.convert('RGBA')
    alpha_rgba = alpha_pil.convert('RGBA')
    alpha_channel = alpha_rgba.split()[3]
    
    # Create final image with smoothed RGB but original alpha
    final_img = Image.new('RGBA', all_rgba.size)
    final_img.paste(all_rgba, (0, 0))
    final_img.putalpha(alpha_channel)
    
    # Apply selective alpha
    final_img = apply_selective_alpha(final_img, alpha_channel, df)
    
    return final_img

def render_predictions_xray_view(df, output_path, width=6000, height=6000):
    """Render tree with X-ray effect and red wood based on predictions"""
    print("Creating predictions X-ray visualization...")
    
    # Calculate data ranges
    x_span = df['x'].max() - df['x'].min()
    y_span = df['y'].max() - df['y'].min()
    z_span = df['z'].max() - df['z'].min()
    
    # Determine which side view is more informative
    use_xz = (x_span * z_span) >= (y_span * z_span)
    print(f"Using {'XZ' if use_xz else 'YZ'} projection for side view")
    
    # Create custom colormap: 0 (leaves) = black, 1 (wood) = red
    pred_cmap = LinearSegmentedColormap.from_list('pred_colors', [
        (0.0, 0.0, 0.0, 1.0),  # Black for leaves
        (1.0, 0.0, 0.0, 1.0),  # Red for wood
    ])
    
    # Side view
    print("\nCreating predictions side view...")
    aspect_ratio = (x_span if use_xz else y_span) / z_span
    canvas, canvas_width, canvas_height = create_canvas(width, height, aspect_ratio)
    
    side_view = render_view(df, canvas, 'x' if use_xz else 'y', 'z', ds.max('prediction'), pred_cmap, (0, 1))
    
    # Composite and save side view
    background = Image.new('RGBA', (canvas_width, canvas_height), color='white')
    side_view = Image.alpha_composite(background, side_view)
    
    # Resize if needed
    if canvas_width > width or canvas_height > height:
        if canvas_width / canvas_height > width / height:
            new_width, new_height = width, int(canvas_height * (width / canvas_width))
        else:
            new_width, new_height = int(canvas_width * (height / canvas_height)), height
        side_view = side_view.resize((new_width, new_height), Image.LANCZOS)
    
    side_view_path = output_path.replace('.png', '_side.png')
    side_view.save(side_view_path, quality=100)
    print(f"Saved predictions side view to {side_view_path}")
    
    # Bottom view (XY)
    print("\nCreating predictions bottom view...")
    aspect_ratio = x_span / y_span
    bottom_canvas, bottom_width, bottom_height = create_canvas(width, height, aspect_ratio)
    
    bottom_view = render_view(df, bottom_canvas, 'x', 'y', ds.max('prediction'), pred_cmap, (0, 1))
    
    # Composite and save bottom view
    bottom_background = Image.new('RGBA', (bottom_width, bottom_height), color='white')
    bottom_view = Image.alpha_composite(bottom_background, bottom_view)
    
    # Resize if needed
    if bottom_width > width or bottom_height > height:
        if bottom_width / bottom_height > width / height:
            new_width, new_height = width, int(bottom_height * (width / bottom_width))
        else:
            new_width, new_height = int(bottom_width * (height / bottom_height)), height
        bottom_view = bottom_view.resize((new_width, new_height), Image.LANCZOS)
    
    bottom_view_path = output_path.replace('.png', '_bottom.png')
    bottom_view.save(bottom_view_path, quality=100)
    print(f"Saved predictions bottom view to {bottom_view_path}")
    
    return side_view, bottom_view

def render_pwood_view(df, output_path, width=6000, height=6000, colormap='fire'):
    """Render tree with X-ray effect and specified colormap for wood based on pwood values"""
    print(f"Creating pwood-colored X-ray views using {colormap} colormap...")
    
    # Calculate data ranges
    x_span = df['x'].max() - df['x'].min()
    y_span = df['y'].max() - df['y'].min()
    z_span = df['z'].max() - df['z'].min()
    
    # Determine which side view is more informative
    use_xz = (x_span * z_span) >= (y_span * z_span)
    print(f"Using {'XZ' if use_xz else 'YZ'} projection for side view")
    
    # Create custom colormap
    if colormap.lower() == 'fire':
        pwood_cmap = create_fire_colormap()
    else:
        pwood_cmap = create_colormap(colormap)
    
    # Create dataframe with pwood values for wood and 0 for leaves
    df_with_pwood = df.copy()
    df_with_pwood['pwood_for_coloring'] = df_with_pwood.apply(
        lambda row: row['pwood'] if row['prediction'] == 1 else 0.0, axis=1
    )
    
    # Get wood pwood range for colormap span
    wood_pwood_values = df_with_pwood[df_with_pwood['prediction'] == 1]['pwood_for_coloring'].values
    if len(wood_pwood_values) > 0:
        span = (wood_pwood_values.min(), wood_pwood_values.max())
        print(f"Wood pwood range: {span[0]:.3f} to {span[1]:.3f}")
    else:
        span = None
        print("Warning: No wood points found")
    
    # Side view
    print("\nCreating pwood side view...")
    aspect_ratio = (x_span if use_xz else y_span) / z_span
    canvas, canvas_width, canvas_height = create_canvas(width, height, aspect_ratio)
    
    side_view = render_view(df_with_pwood, canvas, 'x' if use_xz else 'y', 'z', ds.mean('pwood_for_coloring'), pwood_cmap, span)
    
    # Composite and save side view
    background = Image.new('RGBA', (canvas_width, canvas_height), color='white')
    side_view = Image.alpha_composite(background, side_view)
    
    # Resize if needed
    if canvas_width > width or canvas_height > height:
        if canvas_width / canvas_height > width / height:
            new_width, new_height = width, int(canvas_height * (width / canvas_width))
        else:
            new_width, new_height = int(canvas_width * (height / canvas_height)), height
        side_view = side_view.resize((new_width, new_height), Image.LANCZOS)
    
    side_view_path = output_path.replace('.png', '_side.png')
    side_view.save(side_view_path, quality=100)
    print(f"Saved pwood side view to {side_view_path}")
    
    # Bottom view (XY)
    print("\nCreating pwood bottom view...")
    aspect_ratio = x_span / y_span
    bottom_canvas, bottom_width, bottom_height = create_canvas(width, height, aspect_ratio)
    
    bottom_view = render_view(df_with_pwood, bottom_canvas, 'x', 'y', ds.mean('pwood_for_coloring'), pwood_cmap, span)
    
    # Composite and save bottom view
    bottom_background = Image.new('RGBA', (bottom_width, bottom_height), color='white')
    bottom_view = Image.alpha_composite(bottom_background, bottom_view)
    
    # Resize if needed
    if bottom_width > width or bottom_height > height:
        if bottom_width / bottom_height > width / height:
            new_width, new_height = width, int(bottom_height * (width / bottom_width))
        else:
            new_width, new_height = int(bottom_width * (height / bottom_height)), height
        bottom_view = bottom_view.resize((new_width, new_height), Image.LANCZOS)
    
    bottom_view_path = output_path.replace('.png', '_bottom.png')
    bottom_view.save(bottom_view_path, quality=100)
    print(f"Saved pwood bottom view to {bottom_view_path}")
    
    return side_view, bottom_view

def main():
    # Get input file and colormap from command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        colormap = sys.argv[2] if len(sys.argv) > 2 else 'fire'
    else:
        input_file = input("Enter path to PLY file: ")
        colormap = 'fire'
    
    # Load the PLY file
    print(f"\nLoading PLY file: {input_file}")
    df = load_ply(input_file)
    
    # Create images directory in same location as script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Create output paths
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    predictions_output = os.path.join(images_dir, f"{base_name}_predictions.png")
    pwood_output = os.path.join(images_dir, f"{base_name}_pwood.png")
    
    # Render views
    render_predictions_xray_view(df, predictions_output)
    render_pwood_view(df, pwood_output, colormap=colormap)

if __name__ == "__main__":
    main()