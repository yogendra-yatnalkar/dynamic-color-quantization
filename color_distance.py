import numpy as np

def rgb_color_distance(colors1, colors2, bgr_to_rgb = False):
    """
    Calculates color distances, handling various input types. 
    https://www.compuphase.com/cmetric.htm

    Args:
      colors1: A list or numpy array representing the first set of colors.
      colors2: A list or numpy array representing the second set of colors.

    Returns:
      A numpy array containing the color distances.
    """

    # Convert lists to numpy arrays if necessary
    if isinstance(colors1, list):
      colors1 = np.array(colors1)
    if isinstance(colors2, list):
        colors2 = np.array(colors2)

    # Handle single color inputs
    if colors1.ndim == 1:
        colors1 = colors1.reshape(1, -1)
    if colors2.ndim == 1:
        colors2 = colors2.reshape(1, -1)

    # Invert color if image channels are in BGR format
    if(bgr_to_rgb and isinstance(colors1, np.ndarray) and isinstance(colors2, np.ndarray)):
        colors1 = colors1[..., ::-1]  # Convert from BGR to RGB
        colors2 = colors2[..., ::-1]

    colors1 = colors1.astype(np.float64)
    colors2 = colors2.astype(np.float64)


    r_bar = (colors1[:, 0] + colors2[:, 0]) / 2
    delta_r = colors1[:, 0] - colors2[:, 0]
    delta_g = colors1[:, 1] - colors2[:, 1]
    delta_b = colors1[:, 2] - colors2[:, 2]

    distances = np.sqrt(
        (2 + (r_bar / 256)) * delta_r**2 +
        4 * delta_g**2 +
        (2 + ((255 - r_bar) / 256)) * delta_b**2
    )
    return distances

if __name__ == '__main__':
    # Example usage with Nx3 matrices:
    colors1 = np.array([
        [255, 0, 0],  # Red
        [0, 0, 0],  # Green
        [255, 255, 255],  # Blue
        [255, 255, 255],
        [255, 0, 0] # White
    ], dtype=np.uint8)
    colors2 = np.array([
        [200, 50, 50],  # Slightly different red
        [0, 0, 0],  # Slightly different green
        [255, 240, 210],  # Slightly different blue
        [0, 0, 0],
        [0, 0, 0],
    ], dtype=np.uint8)

    distances = rgb_color_distance(colors1, colors2)
    print("Distances between corresponding colors:")
    print(distances)

    print("######################################")
    # Example usage with Nx3 matrices:
    colors1 = np.array([[255, 255, 255], [0,0,0]], dtype=np.uint8)
    colors2 = [0, 0, 0]

    distances = rgb_color_distance(colors1, colors2)
    print("Distances between corresponding colors:")
    print(distances)