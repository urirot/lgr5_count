
import os
from PIL import Image
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

MIN_CLUSTER_SIZE = 100


def image_to_pixels(image_path="images/Trial.png"):
    """
    Loads an image and converts it to a pixel matrix.

    Args:
        image_path (str): Path to the image file, defaults to "/images/Trial.png"

    Returns:
        numpy.ndarray: Matrix of pixel values
    """
    img = Image.open(image_path)
    # Convert image to RGB if it's not already
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # Convert the image to a numpy array
    pixel_matrix = np.array(img)
    return pixel_matrix


def split_matrix_into_segments(pixel_matrix, num_segments=10):
    """
    Splits the pixel matrix into equal segments along the height.
    
    Args:
        pixel_matrix (numpy.ndarray): The input pixel matrix
        num_segments (int): Number of segments to split into (default: 10)
    
    Returns:
        list: List of numpy arrays, each containing a segment of the image
    """
    height = pixel_matrix.shape[0]
    segment_height = height // num_segments
    segments = []
    
    for i in range(num_segments):
        start_idx = i * segment_height
        end_idx = start_idx + segment_height if i < num_segments - 1 else height
        segment = pixel_matrix[start_idx:end_idx]
        segments.append(segment)
    
    return segments

    
def count_red_clusters(segments, min_cluster_size, red_threshold=150, rgb_difference=50):
    """
    Counts red pixels that are part of clusters with at least min_cluster_size adjacent pixels.
    
    Args:
        segments (list): List of numpy arrays containing image segments
        red_threshold (int): Minimum value for red channel
        rgb_difference (int): Minimum difference between red and other channels
        min_cluster_size (int): Minimum number of adjacent pixels to count as cluster
    
    Returns:
        list: Count of red pixels in valid clusters for each segment
    """
    red_cluster_counts = []
    
    for segment in segments:
        # Extract RGB channels
        red = segment[:, :, 0]
        green = segment[:, :, 1]
        blue = segment[:, :, 2]
        
        # Create mask for red pixels
        red_mask = (red >= red_threshold) & (red > green + rgb_difference) & (red > blue + rgb_difference)
        
        # Find connected components
        labeled_array, num_features = ndimage.label(red_mask)
        
        # Count pixels in each cluster
        cluster_sizes = np.bincount(labeled_array.ravel())
        
        # Create mask for valid clusters (size >= min_cluster_size)
        valid_clusters = cluster_sizes >= min_cluster_size
        valid_clusters[0] = False  # Ignore background (label 0)
        
        # Count pixels in valid clusters
        valid_pixels = np.isin(labeled_array, np.where(valid_clusters)[0])
        red_count = np.sum(valid_pixels)
        
        red_cluster_counts.append(red_count)
    
    return red_cluster_counts


def process_all_images(folder_path="images/control"):
    """
    Process all images in the specified folder and calculate average red clusters per segment.
    
    Args:
        folder_path (str): Path to folder containing images
        
    Returns:
        numpy.ndarray: Array of average red cluster counts per segment position
    """
    # Initialize list to store all results
    all_results = []
    
    # Get all image files from the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        raise ValueError(f"No image files found in {folder_path}")
    
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f"Processing {image_file}...")
        
        # Process single image
        pixels = image_to_pixels(image_path)
        segments = split_matrix_into_segments(pixels)
        red_counts = count_red_clusters(segments, MIN_CLUSTER_SIZE)
        all_results.append(red_counts)
    
    # Convert to numpy array for easier calculations
    results_array = np.array(all_results)
    
    # Calculate averages for each segment position
    segment_averages = np.mean(results_array, axis=0)
    
    return segment_averages


def create_color_gradient_steps(intensities, step_size=50, max_intensity=1800):
    """
    Creates colors for segments based on fixed intensity ranges.
    
    Args:
        intensities (list): List of intensity values
        step_size (int): Size of each intensity step (default: 50)
        max_intensity (int): Maximum intensity value (default: 1800)
    
    Returns:
        dict: Dictionary mapping intensity to color
    """
    start_color = '#f5f2f2'  # light gray
    end_color = '#f70606'    # bright red
    
    # Convert hex to RGB for easier interpolation
    start_rgb = tuple(int(start_color[i:i+2], 16) for i in (1, 3, 5))
    end_rgb = tuple(int(end_color[i:i+2], 16) for i in (1, 3, 5))
    
    intensity_to_color = {}
    
    for intensity in intensities:
        # Calculate which step this intensity belongs to (0-36)
        step = min(36, int(intensity / step_size))
        
        # Linear interpolation between start and end color
        t = step / 36  # Normalize to 0-1 range
        r = int(start_rgb[0] * (1-t) + end_rgb[0] * t)
        g = int(start_rgb[1] * (1-t) + end_rgb[1] * t)
        b = int(start_rgb[2] * (1-t) + end_rgb[2] * t)
        
        intensity_to_color[intensity] = f'#{r:02x}{g:02x}{b:02x}'
    
    return intensity_to_color
    

def plot_segment_intensities(averages, output_filename, intensity_to_color):
    """
    Creates a stacked bar chart with exactly equal-height segments and intensity-based colors.
    """
    # Create figure with fixed dimensions
    fig, ax = plt.subplots(figsize=(2, 10))
    
    # Fixed parameters
    num_segments = 10
    total_height = 1.0
    segment_height = total_height / num_segments
    position = [0]
    
    # Reverse the averages array to plot from bottom to top
    averages = averages[::-1]
    
    # Create stacked bars with exactly equal heights
    for i, value in enumerate(averages):
        bottom = i * segment_height
        ax.bar(position, [segment_height], bottom=bottom,
               color=intensity_to_color[value], width=0.5)
    
    # Remove all decorations
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Set fixed axis limits to ensure equal heights
    ax.set_ylim(0, total_height)
    
    # Save with minimal padding
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
   

def process_and_plot_images(folder_path, output_filename):
    """
    Process images from a folder and create a stacked bar chart.
    
    Args:
        folder_path (str): Path to folder containing images
        output_filename (str): Name of the output PNG file
    """
    # Process images and get averages
    averages = process_all_images(folder_path)
    
    # Create and save plot
    plot_segment_intensities(averages, output_filename)

def print_segment_comparison(control_averages, ko_averages):
    """
    Prints a comparison of segment averages for control and KO groups.
    """
    print("\nSegment Analysis Results:")
    print("=" * 60)
    print(f"{'Segment':<10} {'Control':<15} {'KO':<15}")
    print("-" * 60)
    
    # Reverse arrays to match display order (bottom to top)
    control_averages = control_averages.copy()  # Make copies to avoid modifying originals
    ko_averages = ko_averages.copy()
    
    for i in range(10):
        segment_num = i + 1  # Count from 1 to 10
        print(f"{segment_num:<10} {control_averages[i]:<15.2f} {ko_averages[i]:<15.2f}")
    print("=" * 60)

# Main execution
if __name__ == "__main__":
    try:
        # Get averages for both control and KO
        control_averages = process_all_images("images/control")
        ko_averages = process_all_images("images/ko")
        
        # Combine all intensities and create color mapping
        all_intensities = np.concatenate([control_averages, ko_averages])
        intensity_to_color = create_color_gradient_steps(all_intensities)
        
        # Create plots using the same color mapping
        plot_segment_intensities(control_averages, 
                               "images/output/control_segment_intensities.png",
                               intensity_to_color)
        plot_segment_intensities(ko_averages,
                               "images/output/ko_segment_intensities.png",
                               intensity_to_color)
        
        # Print final comparison
        print_segment_comparison(control_averages, ko_averages)
        
    except Exception as e:
        print(f"Error processing images: {e}")