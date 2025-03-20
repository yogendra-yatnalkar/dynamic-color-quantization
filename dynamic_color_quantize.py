import cv2
import numpy as np
from color_distance import rgb_color_distance
from sklearn.cluster import KMeans
from collections import deque


def img_distance_with_white(img, is_bgr=False):
    """
    # function which takes an np array image as input and computes distance
    # between all unique colors with white color

    Args:
        img: Numpy array (assumes RGB image)
        is_bgr: If image is read using OpenCV then convert it to RGB
                or pass this info to future functions

    Returns:
        distance_with_white_arr: A 1D numpy array which contains all the distances
                white color
    """
    if is_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    white_color = [255, 255, 255]
    unique_colors = np.unique(img.reshape(-1, 3), axis=0)

    # compute distance of each  unique color with white color
    color_distances = rgb_color_distance(unique_colors, white_color)

    return unique_colors, color_distances


def calibrate_array(
    array, min_val=0, max_val=764.83396636, target_min=0, target_max=100
):
    """
    Calibrates a NumPy array to a target range [target_min, target_max]
    based on a provided or default min/max and the original array data.

    Args:
        array: The NumPy array to calibrate.
        min_val: The minimum possible value in the original data range.
        max_val: The maximum possible value in the original data range.
        target_min: The minimum value of the target range.
        target_max: The maximum value of the target range.

    Returns:
        A new NumPy array with values scaled to the target range.
    """

    # Ensure the min and max are not equal and check for invalid arguments
    if min_val == max_val:
        raise ValueError("Min and Max values cannot be equal")
    if target_min >= target_max:
        raise ValueError("target_min must be less than target_max")

    # Linear scaling/normalization with respect to the full range
    calibrated_array = (
        ((array - min_val) / (max_val - min_val)) * (target_max - target_min)
    ) + target_min

    # Clip values within the target range to handle potential numerical inaccuracies
    calibrated_array = np.clip(calibrated_array, target_min, target_max)

    return calibrated_array


def kmeans_cluster_1d_with_max_dist(arr):
    """
    Performs binary clustering using k-means on a 1D NumPy array and
    calculates the maximum distance from cluster centers using absolute differences.

    Args:
        arr: A 1D NumPy array.

    Returns:
      A tuple: (cluster labels, cluster centers,
                max_distances)
                max_distances contains a list with maximum distances from each cluster
    """
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(arr.reshape(-1, 1))
    centers = (
        kmeans.cluster_centers_.flatten()
    )  # flattened since we now use it as scalar

    max_distances = []
    for i in range(2):  # 2 clusters
        cluster_points = arr[labels == i]
        if cluster_points.size > 0:  # Check for empty clusters
            distances = np.abs(cluster_points - centers[i])
            max_dist = np.abs(np.max(cluster_points) - np.min(cluster_points))
            max_distances.append(max_dist)
        else:
            max_distances.append(0)

    return labels, centers, max_distances


def build_unbalanced_tree(arr, mapping_arr, threshold=10, percentile=5, debug_mode = False):
    """Builds an unbalanced binary tree by recursive k-means clustering of a 1D array,
       also handling a corresponding mapping array.

    Args:
      arr: The 1D NumPy array to cluster (shape n x 1).
      mapping_arr: The corresponding mapping array (shape n x 3).
      threshold: The maximum distance threshold for further splitting.

    Returns:
        A nested dictionary representing the tree structure.
    """
    if arr.size == 0:
        return {}

    if np.all(arr == arr[0]):
        return {
            "array": arr.tolist(),
            "mapping_arr": mapping_arr.tolist(),
            "center": None,
            "point_of_focus": None,
            "max_dist": None,
            "children": {},
        }

    root = {
        "array": arr,
        "mapping_arr": mapping_arr,
        "center": None,
        "point_of_focus": None,
        "max_dist": None,
        "children": {},
    }
    queue = deque([root])

    while queue:
        node = queue.popleft()
        current_array = node["array"]
        current_mapping_arr = node["mapping_arr"]

        labels, centers, max_distances = kmeans_cluster_1d_with_max_dist(current_array)

        for i in range(2):
            if max_distances[i] > threshold:
                cluster_points = current_array[labels == i]
                cluster_mapping_arr = current_mapping_arr[labels == i]
                cluster_center = centers[i]
                cluster_max_dist = max_distances[i]

                # Calculate the median distance
                median_distance = np.median(cluster_points)

                # Find the color closest to the median distance
                index_closest_to_median = np.argmin(np.abs(cluster_points - median_distance))
                # point_of_focus = cluster_mapping_arr[index_closest_to_median]

                # Calculate distances from each color to the median
                distances_to_median = np.abs(cluster_points - median_distance)

                # Get the indices of the colors closest to the median (5%)
                num_colors_to_avg = int(len(cluster_points) * (percentile / 100))
                if num_colors_to_avg == 0:
                    num_colors_to_avg = 1 # take a single one to prevent edge case

                closest_indices = np.argsort(distances_to_median)[:num_colors_to_avg]

                # Average the closest colors
                closest_colors = cluster_mapping_arr[closest_indices]
                point_of_focus = np.mean(closest_colors, axis=0).astype(np.uint8)

                # get the color which is closest to the center
                index_closest_to_center = np.argmin(
                    np.abs(cluster_points - cluster_center)
                )
                color_closest_to_center = cluster_mapping_arr[index_closest_to_center]

                child_node = {
                    "array": cluster_points,
                    "mapping_arr": cluster_mapping_arr,
                    "center": cluster_center,
                    "point_of_focus": point_of_focus,
                    "point_from_mapping_closest_to_center": color_closest_to_center,
                    "max_dist": cluster_max_dist,
                    "children": {},
                }
                node["children"][f"child_{i}"] = child_node
                queue.append(child_node)

                if(debug_mode):
                    print("====")
                    print("total cluster points: ", len(cluster_points))
                    print("index_closest_to_median: ", index_closest_to_median)
                    print("Num Colors to Avg: ", num_colors_to_avg)
                    print("closest indices as per median: ", closest_indices)

    return root


def get_leaf_nodes(tree):
    """
    Recursively traverses the tree and returns a list of all leaf nodes.

    Args:
        tree: A nested dictionary representing the tree structure.

    Returns:
        A list of dictionaries, where each dictionary represents a leaf node.
    """
    leaf_nodes = []

    def _traverse(node):
        if not node["children"]:  # if no children, then it is leaf node
            leaf_nodes.append(node)  # add leaf node to the leaf_nodes list
        else:
            for child in node["children"].values():  # iterate through all the children.
                _traverse(child)  # traverse to the child node

    _traverse(tree)  # start the traversal from root of the tree
    return leaf_nodes


def create_color_bars(rgb_colors, height=20, width=200):
    """
    Creates an image with horizontal color bars from a list of RGB colors.

    Args:
        rgb_colors: A list of RGB color values, each represented as a list [R, G, B].
        height: Height of each color bar in pixels.
        width: Width of the image.

    Returns:
        A NumPy array representing the image.
    """
    num_colors = len(rgb_colors)
    img = np.zeros((num_colors * height, width, 3), dtype=np.uint8)

    for i, color in enumerate(rgb_colors):
        img[i * height : (i + 1) * height, :, 0] = color[2]  # Red channel
        img[i * height : (i + 1) * height, :, 1] = color[1]  # Green channel
        img[i * height : (i + 1) * height, :, 2] = color[0]  # Blue channel
    img = img.astype(np.uint8)

    return img


def create_color_bars_with_numbers(
    rgb_colors,
    numbers,
    height=20,
    width=200,
    max_rows_per_column=10,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.5,
    font_thickness=1,
):
    """
    Creates an image with horizontal color bars from a list of RGB colors and displays corresponding numbers,
    arranging into multiple columns if needed.

    Args:
        rgb_colors: A list of RGB color values, each represented as a list [R, G, B].
        numbers: A list of numbers to display corresponding to each color bar.
        height: Height of each color bar in pixels.
        width: Width of each column of the image.
        max_rows_per_column: Maximum number of rows in each column before creating a new column.
        font: OpenCV font to use for displaying numbers.
        font_scale: Font scale factor.
        font_thickness: Thickness of the font.

    Returns:
        A NumPy array representing the image.
    """
    if len(rgb_colors) != len(numbers):
        raise ValueError("The length of rgb_colors and numbers must be the same.")

    num_colors = len(rgb_colors)
    num_columns = (
        num_colors + max_rows_per_column - 1
    ) // max_rows_per_column  # Calculate the number of columns
    img_width = num_columns * width  # Total width of the final image
    img_height = max_rows_per_column * height  # max height

    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    for i, (color, num) in enumerate(zip(rgb_colors, numbers)):

        col_index = i // max_rows_per_column
        row_index = i % max_rows_per_column

        x_start = col_index * width
        x_end = (col_index + 1) * width
        y_start = row_index * height
        y_end = (row_index + 1) * height

        img[y_start:y_end, x_start:x_end, 0] = color[2]  # Red channel
        img[y_start:y_end, x_start:x_end, 1] = color[1]  # Green channel
        img[y_start:y_end, x_start:x_end, 2] = color[0]  # Blue channel

        # Format the number
        formatted_num = f"{(num * 100):.1f}"

        text = str(formatted_num) + " % - " + str(color) 
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = x_start + (width - text_size[0]) // 2
        text_y = y_start + (height + text_size[1]) // 2

        # Add background rectangle for the text
        text_bg_padding = 2  # Small padding around text
        text_bg_x1 = text_x - text_bg_padding
        text_bg_y1 = text_y - text_size[1] - text_bg_padding
        text_bg_x2 = text_x + text_size[0] + text_bg_padding
        text_bg_y2 = text_y + text_bg_padding

        cv2.rectangle(
            img, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), (255, 255, 255), -1
        )  # White background

        cv2.putText(
            img,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness,
            cv2.LINE_AA,
        )  # Black text

    img = img.astype(np.uint8)
    return img


def compute_cluster_strength(color_dist_with_cluster):
    """
    Args:
        color_dist_with_cluster: Each individual colors distance with the each clusters color.
            The cluster color can be either the color closest to the center or the color which
            has the maximum frequency
    """
    color_dist_with_cluster = np.array(color_dist_with_cluster)
    color_dist_with_cluster_argmin = np.argmin(color_dist_with_cluster, axis=0)
    _, cluster_strength = np.unique(color_dist_with_cluster_argmin, return_counts=True)
    cluster_strength = cluster_strength / np.sum(cluster_strength)
    return cluster_strength

def get_cluster_corresponding_image_and_strength(img, unique_cluster_colors):
    #Img shape will be nxmx3. First flatten it to shape: (n*m)x3
    # post that, multiply axis 1 with 1000000 and axis 2 with 1000 and add all 3 dims
    #final shape should be: (n*m)

    img_flat = img.copy()
    img_flat = img_flat.astype(np.int32)
    img_flat = img_flat.reshape(-1, 3)
    img_flat_updated = (img_flat.T[0]*1000000 + img_flat.T[1]*1000 + img_flat.T[2]).T

    # Same post processing to be carried for unique colors array
    # shape input: zx3
    # shape output: z
    unique_cluster_colors = unique_cluster_colors.astype(np.int32)
    unique_cluster_colors_updated = (unique_cluster_colors.T[0]*1000000 + unique_cluster_colors.T[1]*1000 + unique_cluster_colors.T[2]).T

    # now, using np.isin, check which all colors present in the img are also present
    # in the list of unique colors of the cluster
    mask = np.isin(img_flat_updated, unique_cluster_colors_updated)
    cluster_strength = np.sum(mask)/len(mask)

    mask = mask.reshape(img.shape[:2])
    mask = mask*1 # binary mask

    # get all the pixels where mask value == 1
    indices_for_cluster = np.where(mask == 1)

    cluster_img = np.zeros(img.shape)
    cluster_img[indices_for_cluster] = img[indices_for_cluster]

    cluster_img = cluster_img.astype(np.uint8)
    cluster_img = cv2.cvtColor(cluster_img, cv2.COLOR_RGB2BGR)

    return cluster_img, cluster_strength


def dynamic_color_quantize(img, is_bgr=False, threshold=5, percentile=5):
    """
    threshold: The min distance between 2 colors is 0 (distance with itself) and max distance is 100
        as the distances are calibrated between 0 and 100. . The threshold decides, within individual cluster,
        what will be the maximum distance between 2 extreme points.
    """
    # compute colors distance with white
    unique_colors, color_distances = img_distance_with_white(img, is_bgr=is_bgr)
    # print(np.max(color_distances), np.min(color_distances))

    # calibrate the distances to be between 0 and 100
    color_distances = calibrate_array(
        color_distances, min_val=0, max_val=764.83396636, target_min=0, target_max=100
    )
    # print(np.max(color_distances), np.min(color_distances))

    # build the tree with clustering
    tree = build_unbalanced_tree(color_distances, unique_colors, threshold=threshold, percentile=percentile)
    leaf_nodes = get_leaf_nodes(tree)

    # get the quantized colors
    median_clr_in_cluster_li = []
    center_closest_color_in_cluster_li = []

    cluster_strength_li = []
    cluster_images_li = []
    for leaf in leaf_nodes:

        median_clr_in_cluster_li.append(leaf["point_of_focus"])
        center_closest_color_in_cluster_li.append(
            leaf["point_from_mapping_closest_to_center"]
        )

        # get cluster strength
        cluster_img, cluster_strn = get_cluster_corresponding_image_and_strength(img, leaf["mapping_arr"])
        cluster_images_li.append(cluster_img)
        cluster_strength_li.append(cluster_strn)


    return (
        median_clr_in_cluster_li,
        center_closest_color_in_cluster_li,
        cluster_images_li,
        cluster_strength_li,
        leaf_nodes,
    )


if __name__ == "__main__":
    img_path = "./flower.jpg"  # flower or car
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    in_cluster_threshold = 5
    mdian_color_percentile_repr = 0

    (
        median_clr_in_cluster,
        center_closest_color_in_cluster,
        cluster_images_li,
        cluster_strength_li,
        color_clusters,
    ) = dynamic_color_quantize(img, threshold=in_cluster_threshold, percentile=mdian_color_percentile_repr)

    # Sort the lists based on cluster_strength_li
    sorted_data = sorted(zip(
        median_clr_in_cluster,
        center_closest_color_in_cluster,
        cluster_images_li,
        cluster_strength_li,
        color_clusters
    ), key=lambda x: x[3], reverse=True) #x[3] because the cluster strength is at position 3

    # Unpack the sorted data back into separate lists
    median_clr_in_cluster, \
    center_closest_color_in_cluster, \
    cluster_images_li, \
    cluster_strength_li, \
    color_clusters = zip(*sorted_data)

    # Convert the zipped tuples to list
    median_clr_in_cluster = list(median_clr_in_cluster)
    center_closest_color_in_cluster = list(center_closest_color_in_cluster)
    cluster_images_li = list(cluster_images_li)
    cluster_strength_li = list(cluster_strength_li)
    color_clusters = list(color_clusters)
    
    print("No of clusters: ", len(color_clusters))
    print("Leaf Nodes:")

    for i, leaf in enumerate(color_clusters):
        print("Center: ", leaf["center"])
        print("Max Dist: ", leaf["max_dist"])
        print("Cluster Strength: ", cluster_strength_li[i])
        print(
            "Cluster strenght wrt closest color: ",
            cluster_strength_li[i],
        )
        print("Focus Color: ", leaf["point_of_focus"].tolist())
        print("-----------------")

    median_color_bar_img_with_strn = create_color_bars_with_numbers(
        median_clr_in_cluster,
        cluster_strength_li,
        height=30,
        width=400,
        max_rows_per_column=15,
    )

    cntr_closest_color_bar_img_with_strn = create_color_bars_with_numbers(
        center_closest_color_in_cluster,
        cluster_strength_li,
        height=30,
        width=400,
        max_rows_per_column=15,
    )

    img_to_display = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_to_display = cv2.resize(img_to_display, (600, 600))
    cv2.imshow("img", img_to_display)
    cv2.imshow("Cluster Median Color", median_color_bar_img_with_strn)
    cv2.imshow(
        "Cluster Center Color",
        cntr_closest_color_bar_img_with_strn,
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()
