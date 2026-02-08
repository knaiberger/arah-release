import sys
import os
import pickle
import numpy as np
import glob
import torch
import trimesh
import cv2
import time

from scipy.spatial.transform import Rotation
from human_body_prior.body_model.body_model import BodyModel
from preprocess_datasets.easymocap.smplmodel import load_model

# get bone transform and minimal shape as in Arah (ref:https://github.com/taconite/arah-release/blob/main/preprocess_datasets/preprocess_ZJU-MoCap.py)
def get_bone_transform_and_minimal_shape(out_dir,input_dir):

    # create output folder
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # load body model
    body_model = BodyModel(bm_path='body_models/smpl/neutral/model.pkl', num_betas=10, batch_size=1).cpu()
    faces = np.load('body_models/misc/faces.npz')['faces']

    # load model files
    smpl_files = sorted(glob.glob(os.path.join(input_dir, '*.npz')))

    
    for smpl_file in smpl_files:
        ## load params
        params = np.load(smpl_file, allow_pickle=True)
        root_orient = np.array(params['root_orient']).reshape([1, 3]).astype(np.float32)
        trans = np.array(params['trans']).reshape([1, 3])
        #betas shape 1x10
        betas = np.array(params['betas'], dtype=np.float32)
        pose_body = np.array(params['pose_body'], dtype=np.float32).reshape([1, 63])
        pose_hand = np.array(params['pose_hand'], dtype=np.float32).reshape([1, 6])
        poses = np.concatenate((np.array(params['root_orient'], dtype=np.float32).reshape([1,3]),pose_body.copy(),pose_hand.copy()),axis=1)

        ## params to torch
        poses_torch = torch.from_numpy(poses).cuda()
        pose_body_torch = torch.from_numpy(pose_body).cuda()
        pose_hand_torch = torch.from_numpy(pose_hand).cuda()
        betas_torch = torch.from_numpy(betas).cuda()
        trans_torch = torch.from_numpy(trans).cuda()
        root_orient_torch = torch.from_numpy(root_orient).cuda()

        # params for body_model_em
        #rt = Rotation.from_rotvec(np.array(params['root_orient']).reshape([-1])).as_matrix()
        #new_root_orient = Rotation.from_matrix(rt).as_rotvec().reshape([1, 3]).astype(np.float32)
        #new_trans = trans.reshape([1, 3]).astype(np.float32)

        # params for body_model_em to torch 
        #new_root_orient_torch = torch.from_numpy(new_root_orient).cuda()
        #new_trans_torch = torch.from_numpy(new_trans).cuda()

        # Get shape vertices
        body = body_model(betas=betas_torch)
        minimal_shape = body.v.detach().cpu().numpy()[0]

        # Get smpl body for bone transform
        body = body_model(root_orient=root_orient_torch, pose_body=pose_body_torch, pose_hand=pose_hand_torch,
                          betas=betas_torch, trans=trans_torch)

        #body_model_em = load_model(gender='neutral', model_type='smpl')
        #verts = body_model_em(poses=poses_torch, shapes=betas_torch, Rh=new_root_orient_torch, Th=new_trans_torch, return_verts=True)[0].detach().cpu().numpy()

        #vertices = body.v.detach().cpu().numpy()[0]
        #new_trans = new_trans + (verts - vertices).mean(0, keepdims=True)
        #new_trans_torch = torch.from_numpy(new_trans).cuda()

        #body = body_model(root_orient=new_root_orient_torch, pose_body=pose_body_torch, pose_hand=pose_hand_torch, betas=betas_torch, trans=new_trans_torch)

        # get bone transform shape: 1x24x4x4
        bone_transforms = body.bone_transforms.detach().cpu().numpy()
        abs_bone_transforms = body.abs_bone_transforms.detach().cpu().numpy()

        print(np.array(params['trans']).reshape([1, 3]))
        #print(new_trans)
        out_filename = os.path.join(out_dir,smpl_file[-10:])
        print(out_filename)
        #minimal_shape shape: 6890x3x10
        np.savez(out_filename,
                 minimal_shape=minimal_shape,
                 betas=betas,
                 bone_transforms=bone_transforms[0],
                 abs_bone_transforms = abs_bone_transforms[0],
                 trans=trans[0],
                 root_orient=root_orient[0],
                 pose_body=pose_body[0],
                 pose_hand=pose_hand[0])





input_folder = sys.argv[1]
output_folder = sys.argv[2]
get_bone_transform_and_minimal_shape(output_folder,input_folder)


#subjects = ["female-3-casual","female-4-casual","male-3-casual","male-4-casual"]
#subjects = ["female-3-casual"]
#cam_types = ["multiple","first","average"]
#cam_types = ["first"]
#for i in subjects:
 #   for j in cam_types:
        #get_bone_transform(os.path.join("../../data/peoplesnapshot_arah-format/people_snapshot_public/","female-4-casual","camerahmr_foreground"+j),os.path.join("../CameraHMR/output/",i,j,"camerahmr_optimized"))
  #      start = time.time()
   #     get_bone_transform(os.path.join("../../data/peoplesnapshot_arah-format/people_snapshot_public/",i,"camerahmr_"+j),os.path.join("../CameraHMR/output/",i,j,"camerahmr"))
    #    end = time.time()
     #   print("Timer to bring ínto arah format: "+str(end - start))

#get_bone_transform("output")
