'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit.
2. Please Read the instructions and do not modify the input and output formats of function detect_faces() and cluster_faces().
3. If you want to show an image for debugging, please use show_image() function in helper.py.
4. Please do NOT save any intermediate files in your final submission.
'''


import torch

import face_recognition

from typing import Dict, List
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

def detect_faces(img: torch.Tensor) -> List[List[float]]:
    """
    Args:
        img : input image is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).

    Returns:
        detection_results: a python nested list.
            Each element is the detected bounding boxes of the faces (may be more than one faces in one image).
            The format of detected bounding boxes a python list of float with length of 4. It should be formed as
            [topleft-x, topleft-y, box-width, box-height] in pixels.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects.
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.

    """
    detection_results: List[List[float]] = []

    ##### YOUR IMPLEMENTATION STARTS HERE #####

    # img from task1.py is shape (C, H, W) uint8 RGB (torchvision.io.read_image returns C x H x W)
    # face_recognition expects a (H, W, C) uint8 RGB numpy array
    # We use .permute() (torch op) then .numpy() only for the face_recognition API call
    img_hwc = img.permute(1, 2, 0).contiguous()   # (H, W, C)
    img_np = img_hwc.numpy()                        # passed to face_recognition API only

    # Detect face bounding boxes; use HOG model (fast, CPU-friendly)
    # Returns list of (top, right, bottom, left) tuples
    locations = face_recognition.face_locations(img_np, number_of_times_to_upsample=1, model='hog')

    for (top, right, bottom, left) in locations:
        x = float(left)
        y = float(top)
        w = float(right - left)
        h = float(bottom - top)
        detection_results.append([x, y, w, h])

    return detection_results



def cluster_faces(imgs: Dict[str, torch.Tensor], K: int) -> List[List[str]]:
    """
    Args:
        imgs : input images. It is a python dictionary
            The keys of the dictionary are image names (without path).
            Each value of the dictionary is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).
        K: Number of clusters.
    Returns:
        cluster_results: a python list where each elemnts is a python list.
            Each element of the list a still a python list that represents a cluster.
            The elements of cluster list are python strings, which are image filenames (without path).
            Note that, the final filename should be from the input "imgs". Please do not change the filenames.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects.
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.

    """
    cluster_results: List[List[str]] = [[] for _ in range(K)] # Please make sure your output follows this data format.

    ##### YOUR IMPLEMENTATION STARTS HERE #####

    all_names: List[str] = []
    all_encodings: List[torch.Tensor] = []

    for img_name, img in imgs.items():
        # task2.py calls utils.bgr_to_rgb() on images already in RGB (torchvision returns RGB),
        # which effectively flips them to BGR.  We flip channels back to RGB before passing to
        # face_recognition (which expects RGB).  torch.flip is a pure torch op — no new import.
        img_rgb = img.flip(dims=[0])                    # (C, H, W) BGR → RGB
        img_hwc = img_rgb.permute(1, 2, 0).contiguous()  # (H, W, C) RGB
        img_np = img_hwc.numpy()                        # for face_recognition API only

        # Detect face location in this image (each image has exactly one face per spec)
        locations = face_recognition.face_locations(img_np, number_of_times_to_upsample=1, model='hog')

        if not locations:
            # Fallback: try with upsampling to catch harder faces
            locations = face_recognition.face_locations(img_np, number_of_times_to_upsample=2, model='hog')

        # Get 128-dim face encoding using face_recognition API
        # face_encodings expects known_face_locations as list of (top, right, bottom, left)
        if locations:
            face_encs = face_recognition.face_encodings(img_np, locations)
        else:
            face_encs = face_recognition.face_encodings(img_np)

        if face_encs:
            enc = torch.tensor(face_encs[0], dtype=torch.float32)
        else:
            enc = torch.zeros(128, dtype=torch.float32)

        all_names.append(img_name)
        all_encodings.append(enc)

    if not all_names:
        return cluster_results

    # Stack all encodings into a single tensor: N x 128
    enc_tensor = torch.stack(all_encodings)   # (N, 128)
    N = enc_tensor.shape[0]

    # ------------------------------------------------------------------
    # K-Means clustering — implemented from scratch using PyTorch only
    # (No OpenCV, no sklearn, no library cluster APIs allowed)
    # ------------------------------------------------------------------
    best_assignments = _kmeans(enc_tensor, K)

    # Populate cluster result lists
    for i, name in enumerate(all_names):
        cluster_id = int(best_assignments[i].item())
        cluster_results[cluster_id].append(name)

    return cluster_results


'''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''

def _kmeans(enc_tensor: torch.Tensor, K: int, n_init: int = 10, max_iter: int = 300) -> torch.Tensor:
    """
    Custom K-Means using PyTorch only.
    Runs n_init times with different random seeds and returns the assignment
    that yields the lowest total within-cluster sum of squared distances.

    Args:
        enc_tensor: (N, D) float tensor of face encodings
        K:          number of clusters
        n_init:     number of independent runs (picks best result)
        max_iter:   maximum iterations per run

    Returns:
        assignments: (N,) long tensor with cluster index for each sample
    """
    N, D = enc_tensor.shape
    K = min(K, N)   # safety: can't have more clusters than samples

    best_assignments = torch.zeros(N, dtype=torch.long)
    best_inertia = float('inf')

    for run in range(n_init):
        torch.manual_seed(run * 42)

        # K-Means++ initialisation for faster convergence
        centers = _kmeans_plus_plus_init(enc_tensor, K)

        assignments = torch.zeros(N, dtype=torch.long)

        for iteration in range(max_iter):
            # Assign each point to the nearest centre
            # cdist computes pairwise Euclidean distances: (N, K)
            dists = torch.cdist(enc_tensor, centers)
            new_assignments = torch.argmin(dists, dim=1)   # (N,)

            if torch.equal(new_assignments, assignments) and iteration > 0:
                break   # converged
            assignments = new_assignments

            # Update centres as mean of assigned points
            new_centers = torch.zeros_like(centers)
            for k in range(K):
                mask = (assignments == k)
                if mask.sum() > 0:
                    new_centers[k] = enc_tensor[mask].mean(dim=0)
                else:
                    # Empty cluster: re-seed with the point farthest from any centre
                    dists_min, _ = dists.min(dim=1)
                    new_centers[k] = enc_tensor[dists_min.argmax()]
            centers = new_centers

        # Compute inertia (total within-cluster squared distance)
        dists = torch.cdist(enc_tensor, centers)
        min_dists = dists[torch.arange(N), assignments]
        inertia = (min_dists ** 2).sum().item()

        if inertia < best_inertia:
            best_inertia = inertia
            best_assignments = assignments.clone()

    return best_assignments


def _kmeans_plus_plus_init(enc_tensor: torch.Tensor, K: int) -> torch.Tensor:
    """
    K-Means++ initialisation: choose the first centre randomly, then each
    subsequent centre with probability proportional to squared distance
    from the nearest already-chosen centre.

    Returns:
        centers: (K, D) tensor
    """
    N, D = enc_tensor.shape
    centers = torch.zeros(K, D, dtype=enc_tensor.dtype)

    # Pick first centre uniformly at random
    idx = torch.randint(N, (1,)).item()
    centers[0] = enc_tensor[idx]

    for k in range(1, K):
        # Squared distance from each point to nearest existing centre
        current_centers = centers[:k]                      # (k, D)
        dists = torch.cdist(enc_tensor, current_centers)   # (N, k)
        min_dists_sq, _ = dists.min(dim=1)                 # (N,)
        min_dists_sq = min_dists_sq ** 2

        # Sample next centre with probability proportional to min_dists_sq
        total = min_dists_sq.sum()
        if total == 0:
            idx = torch.randint(N, (1,)).item()
        else:
            probs = min_dists_sq / total
            idx = int(torch.multinomial(probs, 1).item())
        centers[k] = enc_tensor[idx]

    return centers
