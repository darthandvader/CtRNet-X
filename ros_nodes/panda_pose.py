#!/usr/bin/env python3
import sys
import os
import warnings
base_dir = os.path.abspath(".")
sys.path.append(base_dir)

import numpy as np
import time
import rospy
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber
import sensor_msgs
import geometry_msgs
import kornia

import torch
import torchvision.transforms as transforms

from PIL import Image as PILImage
from utils import *
from models.CtRNet import CtRNet

import cv2
bridge = CvBridge()

import transforms3d as t3d
import tf2_ros

from ros_nodes.particle_filter import *
from ros_nodes.probability_funcs import *
#os.environ['ROS_MASTER_URI']='http://192.168.1.116:11311'
#os.environ['ROS_IP']='192.168.1.186'

################################################################
import argparse
parser = argparse.ArgumentParser()

args = parser.parse_args("")

args.base_dir = "/home/workspace/src/CtRNet-X/"
args.confidence_threshold = 0.45
args.data_folder = "" # your local base directory

args.use_gpu = True
args.trained_on_multi_gpus = True
args.keypoint_seg_model_path = os.path.join(args.base_dir,"weights/net_epoch_260.pth")
args.urdf_file = os.path.join(args.base_dir,"urdfs/Panda/panda.urdf")
args.robot_name = 'Panda' # "Panda" or "Baxter_left_arm"
args.n_kp = 12
# args.scale = 0.5
# args.height = 480
# args.width = 640
args.scale = 0.15625
args.height = 1536
args.width = 2048
args.fx, args.fy, args.px, args.py = 967.2597045898438, 967.2623291015625, 1024.62451171875, 772.18994140625

# scale the camera parameters
args.width = int(args.width * args.scale)
args.height = int(args.height * args.scale)
args.fx = args.fx * args.scale
args.fy = args.fy * args.scale
args.px = args.px * args.scale
args.py = args.py * args.scale

if args.use_gpu:
    device = "cuda"
else:
    device = "cpu"

trans_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

CtRNet = CtRNet(args)

def preprocess_img(cv_img,args):
    image_pil = PILImage.fromarray(cv_img)
    width, height = image_pil.size
    new_size = (int(width*args.scale),int(height*args.scale))
    image_pil = image_pil.resize(new_size)
    image = trans_to_tensor(image_pil)
    return image


#############################################################################3

#start = time.time()
new_data = False
points_2d = None
joint_angles = None
cTr = None
skipped = False
def gotData(img_msg, joint_msg):
    #global start
    global new_data, points_2d, joint_angles, cTr, skipped
    print("Received data!")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(img_msg, "bgr8")
        cv_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        image = preprocess_img(cv_img,args)

        joint_angles = np.array(joint_msg.position)[[0,1,2,3,4,5,6]]

        points_2d, points_3d, confidence = CtRNet.inference_keypoints_dark(image, joint_angles)
        confidence_threshold = args.confidence_threshold
        confidence_mask = confidence > confidence_threshold
        filtered_points_2d = points_2d[confidence_mask].unsqueeze(0)
        filtered_points_3d = points_3d[confidence_mask.squeeze(0)]

        num_confident_thresh = 6
        num_confident = filtered_points_3d.shape[0]
        if num_confident < num_confident_thresh:
            skipped = True
            print(f"Only {num_confident} points are over {confidence_threshold} confident, skipped!")
            return
        cTr = CtRNet.bpnp(filtered_points_2d, filtered_points_3d, CtRNet.K)
        # cTr, points_2d, segmentation = CtRNet.inference_single_image(image, joint_angles)
        #cTb = CtRNet.bpnp(CtRNet.points_2d_pred, CtRNet.points_3d, CtRNet.K).detach().cpu()

    except CvBridgeError as e:
        print(e)
    new_data = True
def update_publisher(cTr, quaternion=True):
    cvTr= np.eye(4)
    if quaternion:
        cvTr[:3, :3] = kornia.geometry.conversions.quaternion_to_rotation_matrix(cTr[:, :4]).detach().cpu().numpy().squeeze()
        cvTr[:3, 3] = np.array(cTr[:, 4:].detach().cpu())
    else:
        cvTr[:3, :3] = kornia.geometry.conversions.angle_axis_to_rotation_matrix(cTr[:, :3]).detach().cpu().numpy().squeeze()
        cvTr[:3, 3] = np.array(cTr[:, 3:].detach().cpu())

    # ROS camera to CV camera transform
    cTcv = np.array([[0, 0 , 1, 0], [-1, 0, 0 , 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    T = cTcv@cvTr
    qua = t3d.quaternions.mat2quat(T[:3, :3]) # wxyz
    # Publish Transform
    br = tf2_ros.TransformBroadcaster()
    t = geometry_msgs.msg.TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = "camera_base"
    t.child_frame_id = "panda_link0"
    t.transform.translation.x = T[0, 3]
    t.transform.translation.y = T[1, 3]
    t.transform.translation.z = T[2, 3]
    t.transform.rotation.x = qua[1]
    t.transform.rotation.y = qua[2]
    t.transform.rotation.z = qua[3]
    t.transform.rotation.w = qua[0]
    br.sendTransform(t)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="`XYZW` quaternion coefficient order is deprecated and will be removed after > 0.6. Please use `QuaternionCoeffOrder.WXYZ` instead.")
    rospy.init_node('panda_pose')
    # Define your image topic
    image_topic = "/rgb/image_raw"
    robot_joint_topic = "/joint_states"
    robot_pose_topic = "robot_pose"
    # Set up your subscriber and define its callback
    #rospy.Subscriber(image_topic, sensor_msgs.msg.Image, gotData)

    image_sub = Subscriber(image_topic, sensor_msgs.msg.Image)
    robot_j_sub = Subscriber(robot_joint_topic, sensor_msgs.msg.JointState)
    pose_pub = rospy.Publisher(robot_pose_topic, geometry_msgs.msg.PoseStamped, queue_size=1)

    ats = ApproximateTimeSynchronizer([image_sub, robot_j_sub], queue_size=10, slop=5)
    ats.registerCallback(gotData)
    init_std = np.array([
                1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, # ori
                1.0e-3, 1.0e-3, 1.0e-3, # pos
            ])
    pf = ParticleFilter(num_states=7,
                        init_distribution=sample_gaussian,
                        motion_model=additive_gaussian,
                        obs_model=point_feature_obs,
                        num_particles=2500)
    pf.init_filter(init_std)
    rospy.loginfo("Initailized particle filter")

    # Main loop:
    rate = rospy.Rate(30) # 30hz
    prev_cTr = None
    use_particle_filter = False
    while not rospy.is_shutdown():
        try:
            if new_data:
                # print("here")
                # Copy all new data gotData
                # new_image = torch.copy(image)
                new_joint_angles = np.copy(joint_angles)
                new_points_2d = torch.clone(points_2d)
                new_cTr = torch.clone(cTr)
                # new_joint_confidence = torch.clone(joint_confidence)
                new_data = False

                if skipped == True:
                    skipped = False
                    continue
                
                if use_particle_filter == False:
                    print("PARTICLE FILTER TURNED OFF")
                    # pred_qua = kornia.geometry.conversions.angle_axis_to_quaternion(cTr[:,:3]).detach().cpu() # xyzw
                    pred_T = cTr[:,3:].detach().cpu()
                    update_publisher(new_cTr, quaternion=False)
                    # update_publisher(new_cTr, new_image_msg, pred_qua.cpu().detach().numpy().squeeze(), pred_T.cpu().detach().numpy().squeeze())
                    continue

                if prev_cTr is None:
                    prev_cTr = new_cTr
                    continue

                # Skip if not CtRNet not confident in joints
                # joint_confident_thresh = 0
                # num_joint_confident = torch.sum(torch.gt(joint_confidence, 0.95))
                # if num_joint_confident < joint_confident_thresh:
                #     print(f"Only confident with {num_joint_confident} joints, skipping...")
                #     continue

                # Predict Particle filter
                pred_std = np.array([1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4,
                                    2.5e-5, 2.5e-5, 2.5e-5])

                pf.predict(pred_std)

                # Update Particle filter
                cam = None
                gamma = 0.15
                pf.update(new_points_2d, CtRNet, new_joint_angles, cam, prev_cTr, gamma)

                mean_particle = pf.get_mean_particle()
                mean_particle_r = torch.from_numpy(mean_particle[:4])
                mean_particle_t = torch.from_numpy(mean_particle[4:])

                prev_cTr_r = kornia.geometry.conversions.angle_axis_to_quaternion(prev_cTr[:, :3])
                prev_cTr_t = prev_cTr[:, 3:]

                pred_cTr = torch.zeros((1, 7))
                pred_cTr[0, :4] = prev_cTr_r.cpu() + mean_particle_r
                pred_cTr[0, 4:] = prev_cTr_t.cpu() + mean_particle_t

                # pred_qua = kornia.geometry.conversions.angle_axis_to_quaternion(pred_cTr[:,:3]).detach().cpu() # xyzw
                # pred_T = pred_cTr[:,3:].detach().cpu()
                # update_publisher(pred_cTr, new_image_msg, pred_qua.cpu().detach().numpy().squeeze(), pred_T.cpu().detach().numpy().squeeze())
                update_publisher(pred_cTr)
            rate.sleep()
        except KeyboardInterrupt:
            # Processing the final video frames in case video length is not a multiple of model.step
            rospy.signal_shutdown("Done.")