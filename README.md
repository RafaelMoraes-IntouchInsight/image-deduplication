# Image Deduplication

<div style="text-align:center">
    <img src="https://raw.githubusercontent.com/farrael004/image-deduplication/main/media/small_logo.png" alt="drawing" style="width:200px;"/>
</div>

Easily find images that have a high degree of similarity using either:
- one line of code
- cli tool

## Installation
```
pip install image-deduplication
```

## Usage

### Find similar images in a folder tree
```python
from image_deduplication import get_image_paths, cluster_images

image_paths = get_image_paths("path/to/images") # Returns list of image paths
image_clusters = cluster_images(image_paths)

# Print clusters
for i, cluster in enumerate(image_clusters):
    print(f"""Cluster {i}: {cluster}\n""")
```

### Find similar images from a list of files
```python
from image_deduplication import cluster_images

image_paths = [
    "image1.png",
    "image2.jpg",
    "image3.jpeg"
]

image_clusters = cluster_images(image_paths)

# Print clusters
for i, cluster in enumerate(image_clusters):
    print(f"""Cluster {i}: {cluster}\n""")
```

### Detecting mirrored duplicates
ORB descriptors are not mirror invariant, so a horizontally flipped copy of an
image scores as unrelated by default. Pass `detect_mirrored=True` to also match
each image against the mirror of the others:

```python
image_clusters = cluster_images(image_paths, detect_mirrored=True)
```

This roughly doubles the pairwise matching work, so it is off by default. On a
set of real photographs it takes flipped-duplicate recall from 0% to 100%
(including flipped *and* cropped copies) without introducing false positives.

### CLI tool
```bash
image-deduplication path/to/images
```
If you want to analyse the current working directory, you can simply use "." as the path.

Add `--detect-mirrored` to also catch horizontally flipped duplicates:
```bash
image-deduplication path/to/images --detect-mirrored
```

## Methodology

Here is an overview of how this package clusters images by similarity using computer vision techniques and a union-find algorithm to group similar images together:

1. Feature Extraction with ORB: We utilized the ORB (Oriented FAST and Rotated BRIEF) algorithm for extracting keypoints and descriptors from images. ORB is a fast, rotation-invariant, and robust feature extractor that identifies unique points in images, facilitating the comparison of different images based on their content. ORB is rotation invariant but *not* mirror invariant, which is why detecting flipped duplicates requires re-extracting features from the mirrored image (see `detect_mirrored`).

2. Image Matching: To determine the similarity between pairs of images, we employed a brute force matcher with the Hamming distance as a metric, optimized to find the best matches for the ORB descriptors. A ratio test filters out less reliable matches, ensuring that only the most similar keypoints contribute to the similarity score.

3. Clustering with Union-Find: An implementation of the union-find algorithm is used to dynamically cluster images based on their similarity scores. This method efficiently merges images into groups as it iterates through all pairs, identifying connected components within the dataset. The union-find structure is key for minimizing redundant comparisons and accelerating the clustering process.

4. Homography and RANSAC: For images with a sufficient number of good matches, we calculate a homography matrix using RANSAC (Random Sample Consensus). This step further refines the matching process by considering only geometrically consistent matches, enhancing the accuracy of similarity scores.

5. Scalable Image Clustering: The end-to-end process, from feature extraction to dynamic clustering, is designed to handle a large number of images. By efficiently comparing images and grouping them based on visual similarity, this package can organize vast datasets into manageable clusters.

This technology approach combines classic computer vision techniques with modern algorithmic strategies, offering a robust solution for organizing and analyzing large collections of images based on their visual content.