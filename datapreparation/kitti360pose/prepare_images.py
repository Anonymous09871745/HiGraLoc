"""Module to extract original (real-world, non-point-cloud) images and corresponding poses out of Kitti360
"""

from typing import List

import os
import os.path as osp
import numpy as np
import pickle
import sys
import time

from shutil import copyfile

from scipy.spatial.distance import cdist


def sample_poses(path_poses, pose_distance):
    poses = np.loadtxt(path_poses)

    image_names = np.intp(poses[:, 0])

    orientations = poses[:, 1:].reshape((-1, 3, 4))
    orientations = orientations[:, 0:3, 0:3]  # Take 3x3 matrices

    poses = poses[:, 1:].reshape((-1, 3, 4))  # Convert to 3x4 matrices
    poses = poses[:, :, -1]  # Take last column

    sampled_poses = [
        poses[0],
    ]
    sampled_orientations = [
        orientations[0],
    ]
    sampled_image_names = [
        image_names[0],
    ]

    for pose, orientation, image_name in zip(poses, orientations, image_names):
        dists = np.linalg.norm(pose - sampled_poses, axis=1)
        if np.min(dists) >= pose_distance:
            sampled_poses.append(pose)
            sampled_orientations.append(orientation)
            sampled_image_names.append(image_name)

    return np.array(sampled_poses), np.array(sampled_orientations), np.array(sampled_image_names)


def create_poses_and_images(path_poses, path_images, path_out, db_dist=20, query_dist=5, step=4):
    poses = np.loadtxt(path_poses)
    image_names = np.intp(poses[:, 0])

    path_db = osp.join(path_out, "real", "db")
    path_query = osp.join(path_out, "real", "query")
    os.makedirs(path_db)
    os.makedirs(path_query)

    poses = poses[:, 1:].reshape((-1, 3, 4))  # Convert to 3x4 matrices
    poses = poses[:, :, -1]  # Take last column

    sampled_db_poses = [
        poses[0],
    ]
    image_name = image_names[0]
    copyfile(
        osp.join(path_images, f"{image_name:010.0f}.png"),
        osp.join(path_db, f"{len(sampled_db_poses) - 1:04.0f}.png"),
    )

    sampled_query_poses = []
    for idx in range(0, len(poses), step):
        pose, image_name = poses[idx], image_names[idx]

        dists = np.linalg.norm(
            pose - sampled_db_poses, axis=1
        )  # Distance to already sampled DB poses
        if np.min(dists) >= db_dist:
            sampled_db_poses.append(pose)
            copyfile(
                osp.join(path_images, f"{image_name:010.0f}.png"),
                osp.join(path_db, f"{len(sampled_db_poses) - 1:04.0f}.png"),
            )
        elif np.min(dists) >= query_dist:
            sampled_query_poses.append(pose)
            copyfile(
                osp.join(path_images, f"{image_name:010.0f}.png"),
                osp.join(path_query, f"{len(sampled_query_poses) - 1:04.0f}.png"),
            )

    with open(osp.join(path_out, "poses_db.pkl"), "wb") as f:
        pickle.dump(np.array(sampled_db_poses), f)
    with open(osp.join(path_out, "poses_query.pkl"), "wb") as f:
        pickle.dump(np.array(sampled_query_poses), f)

    print(f"Saved {len(sampled_db_poses)} / {len(sampled_query_poses)} poses.")

