import os
import cv2
import numpy as np

# Supported image extensions by cv2.imread
supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

def _find_group(groups, image_name):
    if groups[image_name] != image_name:
        groups[image_name] = _find_group(groups, groups[image_name])
    return groups[image_name]

def _union(groups, image_name1, image_name2):
    group1 = _find_group(groups, image_name1)
    group2 = _find_group(groups, image_name2)
    if group1 != group2:
        groups[group2] = group1

def _extract_orb_features(image: np.ndarray):
    """
    Function to extract ORB features from an image.

    Parameters:
    - image: np.ndarray, the input image

    Returns:
    - keypoints: list of keypoints
    - descriptors: list of descriptors
    """
    # Initialize ORB
    orb = cv2.ORB_create()

    # Find the keypoints with ORB
    keypoints = orb.detect(image, None)

    # Compute the descriptors with ORB
    keypoints, descriptors = orb.compute(image, keypoints)

    return keypoints, descriptors

def _match_images(query_img_desc, idx_desc):
    """
    A function to match images using their descriptors and return a similarity score.
    
    Parameters:
    - query_img_desc: The descriptor of the query image
    - idx_desc: The descriptor of the index image
    
    Returns:
    - similarity_score: An integer representing the similarity score between the two images
    """
    # Initialize BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Use BFMatcher to find the best matches
    matches = bf.knnMatch(query_img_desc[1], idx_desc[1], k=2)

    # Apply ratio test to find good matches
    good_matches = [m for m, n in matches if m.distance < 0.75*n.distance]

    # We need at least 50 matches to apply Homography
    if len(good_matches) > 50:
        # Prepare data for cv2.findHomography
        src_pts = np.float32([query_img_desc[0][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([idx_desc[0][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Apply RANSAC
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # See how many matches survive after applying RANSAC
        matches_mask = mask.ravel().tolist()
        num_good_matches = matches_mask.count(1)

        # Use this as a similarity score
        similarity_score = num_good_matches
    else:
        similarity_score = 0  # not enough matches

    return similarity_score

def get_image_paths(folder_path: str, depth: int=None, sanity_check: bool=False) -> list[str]:
    """
    Get the paths of readable images within the specified folder.

    Args:
        folder_path (str): The path of the folder to search for images.
        depth (int, optional): The maximum depth to search for images. Defaults to None.
        sanity_check (bool, optional): Whether to perform sanity checks when reading images. Defaults to False.

    Returns:
        list[str]: A list of paths to readable images within the folder.
    """

    # List to hold the paths of images that can be read
    readable_image_paths = []

    if depth is None:
        depth = float('inf')

    current_depth = 0
    # Iterate over each item in the folder
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)

        # If item is a file and has a supported extension, try reading it
        if os.path.isfile(item_path) and os.path.splitext(item_path)[1].lower() in supported_extensions:
            try:
                if sanity_check:
                    img = cv2.imread(item_path)
                    if img is not None:
                        readable_image_paths.append(item_path)
                else: readable_image_paths.append(item_path)
            except Exception as e:
                print(f"Error reading {item_path}: {e}")
        
        # If item is a directory and the search depth allows, recurse into it
        elif os.path.isdir(item_path) and current_depth < depth:
            readable_image_paths += get_image_paths(item_path, depth, current_depth + 1)

    return readable_image_paths

def cluster_images(image_paths: list[str]) -> list:
    """
    Cluster the given images based on similarity and return a list of grouped images.
    
    Parameters:
    image_paths (list): A list of file paths to the images to be clustered.
    
    Returns:
    list: A list where the each value is a list of image file paths that correspond to grouped images.
    """

    # Assert all images are supported
    for image_path in image_paths:
        ext = os.path.splitext(image_path)[1].lower()
        assert ext in supported_extensions, f"The only file types supported are: {supported_extensions}. Attempted to read unsupported format: {ext}"

    indexed_images = [cv2.imread(name) for name in image_paths]

    # Create descriptors for indexed images
    indexed_img_descs = [_extract_orb_features(image) for image in indexed_images]

    # Initialize a dictionary to store similarity groups
    similarity_groups = {name: name for name in image_paths}
    num_images = len(indexed_img_descs)
    # Compare all images for similarity and update groups
    for i in range(num_images):
        for j in range(i + 1, num_images):
            similarity = _match_images(indexed_img_descs[i], indexed_img_descs[j])
            if similarity > 10:  # Adjust the similarity threshold as needed
                _union(similarity_groups, image_paths[i], image_paths[j])

    # Convert the similarity_groups dictionary into a format where each group is a list
    intermediate_groups = {}
    for image_name, group_name in similarity_groups.items():
        group_root = _find_group(similarity_groups, image_name)
        intermediate_groups.setdefault(group_root, []).append(image_name)

    # Filter out groups with only one image and convert to a list of lists
    final_groups = [group for group in intermediate_groups.values() if len(group) > 1]

    return final_groups