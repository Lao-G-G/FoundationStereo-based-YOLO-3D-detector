#!/usr/bin/env python3
"""
YOLO-3D with FoundationStereo depth estimation for KITTI stereo dataset.

Input: KITTI stereo image pairs (left/right directories or glob pattern)
Output: 3D bounding box visualization with metric depth

Compared to run.py (monocular Depth Anything v2):
- Depth estimation replaced by FoundationStereo (stereo)
- Depth is metric (meters) instead of normalized 0-1
- Input is stereo pairs instead of single camera/video
"""
import os
import sys
import argparse
import time
import cv2
import numpy as np
import torch
from pathlib import Path
from glob import glob

# Set MPS fallback for operations not supported on Apple Silicon
if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Add parent path for FoundationStereo imports
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(code_dir, '..'))

from detection_model import ObjectDetector
from stereo_depth_model import StereoDepthEstimator
from bbox3d_utils import BBox3DEstimator, BirdEyeView
from load_camera_params import load_camera_params


def load_kitti_stereo_pairs(left_dir, right_dir, max_frames=None):
    """
    Find matching stereo pairs from KITTI left/right image directories.

    KITTI naming convention:
    - Left:  image_2/000000.png (color camera 2)
    - Right: image_3/000000.png (color camera 3)
    Or grayscale:
    - Left:  image_0/000000.png
    - Right: image_1/000000.png

    Returns list of (left_path, right_path) tuples.
    """
    # Try common KITTI image patterns
    left_patterns = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    left_files = []
    for pattern in left_patterns:
        left_files.extend(glob(os.path.join(left_dir, pattern)))
    left_files = sorted(left_files)

    right_files = []
    for pattern in left_patterns:
        right_files.extend(glob(os.path.join(right_dir, pattern)))
    right_files = sorted(right_files)

    if len(left_files) == 0:
        raise FileNotFoundError(f"No images found in left directory: {left_dir}")
    if len(right_files) == 0:
        raise FileNotFoundError(f"No images found in right directory: {right_dir}")

    # Match by filename
    pairs = []
    right_basenames = {os.path.basename(f): f for f in right_files}

    for left_file in left_files:
        basename = os.path.basename(left_file)
        if basename in right_basenames:
            pairs.append((left_file, right_basenames[basename]))

    if len(pairs) == 0:
        # Try matching by index if names don't match
        print("[Warning] No exact filename matches found, trying numerical matching...")
        import re
        left_base_to_path = {}
        for f in left_files:
            name = os.path.splitext(os.path.basename(f))[0]
            left_base_to_path[name] = f
        right_base_to_path = {}
        for f in right_files:
            name = os.path.splitext(os.path.basename(f))[0]
            right_base_to_path[name] = f

        common = sorted(set(left_base_to_path.keys()) & set(right_base_to_path.keys()))
        pairs = [(left_base_to_path[k], right_base_to_path[k]) for k in common]

    print(f"Found {len(pairs)} stereo pairs")
    if max_frames is not None:
        pairs = pairs[:max_frames]
        print(f"Limiting to {max_frames} frames")

    return pairs


def main():
    parser = argparse.ArgumentParser(description="YOLO-3D with FoundationStereo for KITTI data")
    parser.add_argument('--left_dir', type=str, 
                        default=r'D:\PyCharm_2025.1.1.1\project\YOLOStereo3D-master\dataset\2011_09_26_drive_0017_sync\2011_09_26\2011_09_26_drive_0017_sync\image_02\data',
                        help='Directory containing left (camera 2) images')
    parser.add_argument('--right_dir', type=str, 
                        default=r'D:\PyCharm_2025.1.1.1\project\YOLOStereo3D-master\dataset\2011_09_26_drive_0017_sync\2011_09_26\2011_09_26_drive_0017_sync\image_03\data',
                        help='Directory containing right (camera 3) images')
    parser.add_argument('--ckpt_dir', type=str,
                        default=os.path.join(code_dir, '..', 'pretrained_models', '11-33-40', 'model_best_bp2.pth'),
                        help='Path to FoundationStereo checkpoint (.pth)')
    parser.add_argument('--output_path', type=str, default='output_stereo.mp4',
                        help='Output video file')
    parser.add_argument('--camera_params', type=str, default=None,
                        help='Path to camera parameters JSON file')
    parser.add_argument('--baseline', type=float, default=0.54,
                        help='Stereo baseline in meters (KITTI default: 0.54m)')
    parser.add_argument('--yolo_model', type=str, default='nano',
                        choices=['nano', 'small', 'medium', 'large', 'extra'],
                        help='YOLOv11 model size')
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                        help='Detection confidence threshold')
    parser.add_argument('--iou_threshold', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Image downsample scale (<=1.0)')
    parser.add_argument('--hiera', action='store_true',
                        help='Use hierarchical inference for high-res images')
    parser.add_argument('--valid_iters', type=int, default=32,
                        help='Number of GRU iterations')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum number of frames to process')
    parser.add_argument('--no_tracking', action='store_true',
                        help='Disable object tracking')
    parser.add_argument('--no_bev', action='store_true',
                        help='Disable Bird\'s Eye View')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda, cpu)')
    parser.add_argument('--z_far', type=float, default=80.0,
                        help='Maximum depth for BEV visualization (meters)')

    args = parser.parse_args()

    # Device setup
    if args.device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load camera parameters
    camera_params = load_camera_params(args.camera_params) if args.camera_params else None

    # Initialize YOLO detector
    print("Initializing YOLO detector...")
    try:
        detector = ObjectDetector(
            model_size=args.yolo_model,
            conf_thres=args.conf_threshold,
            iou_thres=args.iou_threshold,
            classes=None,
            device=device
        )
    except Exception as e:
        print(f"Error initializing detector: {e}")
        detector = ObjectDetector(
            model_size=args.yolo_model,
            conf_thres=args.conf_threshold,
            iou_thres=args.iou_threshold,
            classes=None,
            device='cpu'
        )

    # Initialize FoundationStereo depth estimator
    print("Initializing FoundationStereo depth estimator...")
    # Extract camera intrinsics if available from camera params
    camera_K = None
    if camera_params and 'camera_matrix' in camera_params:
        camera_K = np.array(camera_params['camera_matrix'])

    depth_estimator = StereoDepthEstimator(
        ckpt_dir=args.ckpt_dir,
        camera_K=camera_K,
        baseline=args.baseline,
        device=str(device),
        scale=args.scale,
        hiera=args.hiera,
        valid_iters=args.valid_iters
    )

    # Initialize 3D bbox estimator with camera parameters
    if camera_params:
        camera_matrix = np.array(camera_params.get('camera_matrix', camera_K))
        projection_matrix = np.array(camera_params.get('projection_matrix', camera_matrix))
    else:
        # Use KITTI default from stereo_depth_model
        camera_matrix = depth_estimator.K
        projection_matrix = np.hstack([camera_matrix, np.zeros((3, 1))])

    bbox3d_estimator = BBox3DEstimator(
        camera_matrix=camera_matrix,
        projection_matrix=projection_matrix,
        use_metric_depth=True  # Enable metric depth mode
    )

    # Initialize BEV
    bev = None
    if not args.no_bev:
        bev = BirdEyeView(scale=60, size=(300, 300), use_metric_depth=True)

    # Find stereo pairs
    pairs = load_kitti_stereo_pairs(args.left_dir, args.right_dir, args.max_frames)
    if len(pairs) == 0:
        print("Error: No stereo pairs found!")
        return

    # Read first frame to get dimensions
    sample_img = cv2.imread(pairs[0][0])
    if sample_img is None:
        print(f"Error: Cannot read {pairs[0][0]}")
        return
    height, width = sample_img.shape[:2]

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_path, fourcc, 10.0, (width, height))

    # FPS tracking
    frame_count = 0
    start_time = time.time()
    fps_display = "FPS: --"

    print(f"Processing {len(pairs)} stereo pairs...")
    print("Press 'q' or ESC to quit.")

    for idx, (left_path, right_path) in enumerate(pairs):
        # Check for quit
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            print("Exiting...")
            break

        try:
            # Read stereo pair
            left_img = cv2.imread(left_path)
            right_img = cv2.imread(right_path)
            if left_img is None or right_img is None:
                print(f"Warning: Cannot read pair {idx}: {left_path}, {right_path}")
                continue

            # Resize if needed
            if args.scale < 1.0:
                left_img = cv2.resize(left_img, None, fx=args.scale, fy=args.scale)
                right_img = cv2.resize(right_img, None, fx=args.scale, fy=args.scale)
                height, width = left_img.shape[:2]

            result_frame = left_img.copy()

            t_start = time.time()

            # Step 1: Object Detection (on left image only)
            try:
                _, detections = detector.detect(left_img, track=not args.no_tracking)
            except Exception as e:
                print(f"Detection error at frame {idx}: {e}")
                detections = []
            t_det = time.time()

            # Step 2: Stereo Depth Estimation
            try:
                depth_meters = depth_estimator.estimate_depth(left_img, right_img)
                depth_colored = depth_estimator.colorize_depth(depth_meters)
            except Exception as e:
                print(f"Depth estimation error at frame {idx}: {e}")
                depth_meters = np.zeros((height, width), dtype=np.float32)
                depth_colored = np.zeros((height, width, 3), dtype=np.uint8)
            t_depth = time.time()

            # Step 3: Build 3D boxes with metric depth
            boxes_3d = []
            active_ids = []

            for detection in detections:
                try:
                    bbox, score, class_id, obj_id = detection
                    class_name = detector.get_class_names()[class_id]

                    # Get metric depth in the bounding box region
                    # For people/animals, use center point; otherwise median
                    if class_name.lower() in ['person', 'cat', 'dog']:
                        center_x = int((bbox[0] + bbox[2]) / 2)
                        center_y = int((bbox[1] + bbox[3]) / 2)
                        depth_value = depth_estimator.get_depth_at_point(depth_meters, center_x, center_y)
                        depth_method = 'center_metric'
                    else:
                        depth_value = depth_estimator.get_depth_in_region(depth_meters, bbox, method='median')
                        depth_method = 'median_metric'

                    # Skip if depth is invalid
                    if depth_value <= 0:
                        continue

                    # Estimate full 3D box using metric depth
                    box_3d = bbox3d_estimator.estimate_3d_box(
                        bbox, depth_value, class_name, object_id=obj_id
                    )
                    box_3d['depth_value_metric'] = depth_value
                    box_3d['depth_method'] = depth_method
                    boxes_3d.append(box_3d)

                    if obj_id is not None:
                        active_ids.append(obj_id)
                except Exception as e:
                    print(f"3D box error at frame {idx}: {e}")
                    continue

            # Cleanup trackers
            bbox3d_estimator.cleanup_trackers(active_ids)

            # Step 4: Draw 3D boxes
            for box_3d in boxes_3d:
                try:
                    class_name = box_3d['class_name'].lower()
                    if 'car' in class_name or 'vehicle' in class_name:
                        color = (0, 0, 255)
                    elif 'person' in class_name:
                        color = (0, 255, 0)
                    elif 'bicycle' in class_name or 'motorcycle' in class_name:
                        color = (255, 0, 0)
                    elif 'potted plant' in class_name or 'plant' in class_name:
                        color = (0, 255, 255)
                    else:
                        color = (255, 255, 255)

                    result_frame = bbox3d_estimator.draw_box_3d(result_frame, box_3d, color=color)
                except Exception as e:
                    print(f"Draw error at frame {idx}: {e}")

            # Step 5: BEV visualization
            if not args.no_bev and len(boxes_3d) > 0:
                try:
                    bev.reset()
                    for box_3d in boxes_3d:
                        bev.draw_box(box_3d)
                    bev_image = bev.get_image()

                    bev_height = height // 4
                    bev_width = bev_height
                    bev_resized = cv2.resize(bev_image, (bev_width, bev_height))
                    result_frame[height - bev_height:height, 0:bev_width] = bev_resized
                    cv2.rectangle(result_frame,
                                  (0, height - bev_height),
                                  (bev_width, height),
                                  (255, 255, 255), 1)
                    cv2.putText(result_frame, "Bird's Eye View",
                                (10, height - bev_height + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                except Exception as e:
                    print(f"BEV error at frame {idx}: {e}")

            # Timing and FPS
            t_end = time.time()
            frame_time = t_end - t_start
            frame_count += 1

            # Compute average FPS
            if frame_count == 1:
                avg_fps = 1.0 / max(frame_time, 0.001)
            else:
                elapsed_total = t_end - start_time
                avg_fps = frame_count / max(elapsed_total, 0.001)
            fps_display = f"FPS: {avg_fps:.1f}"

            # Per-step timing (ms)
            det_ms = (t_det - t_start) * 1000
            depth_ms = (t_depth - t_det) * 1000
            draw_ms = (t_end - t_depth) * 1000

            # Console timing every frame
            print(f"Frame {idx+1:04d} | {1000*frame_time:5.0f}ms total | "
                  f"Det:{det_ms:4.0f}ms Depth:{depth_ms:5.0f}ms Draw:{draw_ms:3.0f}ms | "
                  f"Avg {avg_fps:.1f} FPS", end='\r')

            # Add depth map overlay (top-left corner)
            try:
                dh = height // 4
                dw = int(dh * width / height)
                depth_resized = cv2.resize(depth_colored, (dw, dh))
                result_frame[0:dh, 0:dw] = depth_resized
            except Exception:
                pass

            # Overlay info text
            cv2.putText(result_frame, f"{fps_display} | {frame_time*1000:.0f}ms/frame | Frame: {idx + 1}/{len(pairs)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(result_frame, f"D:{det_ms:.0f}ms DP:{depth_ms:.0f}ms | {device}",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Write output
            out.write(result_frame)

            # Display
            cv2.imshow("YOLO-3D Stereo", result_frame)
            cv2.imshow("Stereo Depth", depth_colored)

        except Exception as e:
            print(f"Frame {idx} error: {e}")
            continue

    # Cleanup
    print("Cleaning up...")
    out.release()
    cv2.destroyAllWindows()
    print(f"Done. Output saved to {args.output_path}")
    print(f"Processed {frame_count} frames.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        cv2.destroyAllWindows()