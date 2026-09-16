import argparse
import json
from image_deduplication import get_image_paths, cluster_images

def main():
    parser = argparse.ArgumentParser(description='Find similar images.')
    parser.add_argument('image_folder', type=str, help='Path to the image folder')
    parser.add_argument(
        '--detect-mirrored',
        action='store_true',
        help='Also detect horizontally mirrored duplicates. Roughly doubles the '
             'matching work.',
    )
    
    args = parser.parse_args()

    image_paths = get_image_paths(args.image_folder) # Returns list of image paths
    image_clusters = cluster_images(image_paths, detect_mirrored=args.detect_mirrored)

    print(json.dumps(image_clusters, indent=4))