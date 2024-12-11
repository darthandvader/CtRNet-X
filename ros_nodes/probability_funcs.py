import numpy as np
import torch
import kornia
import cv2
from scipy import optimize
from scipy.stats import norm
from sklearn.metrics.pairwise import rbf_kernel


def sample_gaussian(std):
    m = np.zeros(std.shape)
    sample = norm.rvs(m, std)
    return sample

def additive_gaussian(state, std):
    sample = sample_gaussian(std)
    return state + sample

# def axisAngleToRotationMatrix(axis_angle):
#     # returns (num_particles, 3, 3)
#     angle = np.linalg.norm(axis_angle, keepdims=True, axis=1)
#     # print(angle)
    
#     # if angle < 1e-5:
#     #     return np.eye(3)
    
#     axis = axis_angle/angle
#     print(axis)
#     cross_product_mat_axis = np.array([[0, -axis[2], axis[1]],
#                                        [axis[2], 0, -axis[0]],
#                                        [-axis[1], axis[0], 0]])
#     return np.cos(angle)*np.eye(3) + np.sin(angle) * cross_product_mat_axis \
#             + (1.0 - np.cos(angle))*np.outer(axis,axis)

def projectPoints(points, K):
    # points is Nx7x3 np array
    # print(points)
    num_particles = points.shape[0]
    K = np.tile(K, (num_particles, 1, 1))
    points = points.transpose((0,2,1))
    dehomog_pts = points / np.expand_dims(points[:,-1,:], 1)
    # print(f"HERE {dehomog_pts}")
    projected_point = np.matmul(K, dehomog_pts).transpose((0,2,1))[:,:,:-1]
    # print(projected_point[0])
    
    # points_homogeneous = np.concatenate( (points, np.ones((1, points.shape[1]))) )
    # points_r = np.dot(self.T, points_homogeneous)[:-1, :]
    
    
    return projected_point
# State is: [ori_x, ori_y, ori_z, pos_x, pos_y, pos_z]
def point_feature_obs(states, points_2d, ctrnet, joint_angles, cam, cTr, gamma, class_result):
    print(points_2d.shape)
    #convert state to angle axis
    num_particles = states.shape[0]
    T = np.eye(4)
    T = np.tile(T, (num_particles, 1, 1))

    T[:, :-1, -1] = states[:, 4:]
    # print(axisAngleToRotationMatrix(states[:, :3]).shape)
    
    T[:, :-1, :-1] = kornia.geometry.conversions.quaternion_to_rotation_matrix(torch.from_numpy(states[:, :4]))
    # T[:, :-1, :-1] = axisAngleToRotationMatrix(states[:, :3])
    # Want to have final result of project image plane coords
    # _, t_list = ctrnet.robot.get_joint_RT(joint_angles)
    # p_t = t_list[[0,2,3,4,6,7,8]] # remove 1 and 5 links as they are overlapping with 2 and 6
    p_t = ctrnet.robot.get_3d_keypoints(joint_angles)
    K = np.float64(ctrnet.intrinsics)
    # only projects from camera coords to image coords
    cTr = cTr.cpu().detach().numpy().squeeze()
    cTr_mat = np.eye(4)
    cTr_mat[:-1, -1] = np.array(cTr[3:])
    cTr_mat[:-1, :-1] = kornia.geometry.conversions.angle_axis_to_rotation_matrix(torch.from_numpy(np.expand_dims(cTr[:3], 0)))
    # cTr_mat[:-1, :-1] = axisAngleToRotationMatrix(cTr[:3])
    cTr_mat = np.tile(cTr_mat, (num_particles, 1, 1))
    p_cTr  = cTr_mat @ T
    rvec = np.float64(p_cTr[0, :3, :3])
    tvec = np.float64(p_cTr[0, :3, 3:])
    # print(rvec)
    # print(tvec)
    np.set_printoptions(suppress=True)
    p_c_1 = np.matmul(cTr_mat, T) 
    p_c_2 = np.tile(np.transpose(np.concatenate((p_t, np.ones((p_t.shape[0], 1))), axis=1)), (num_particles, 1, 1))
    # print(f"p_c_1 here: {p_c_1[0]}")
    # print(f"p_c_2 here: {p_c_2[0]}")
    # print(p_c_1.shape, p_c_2.shape)
    p_c = np.matmul(p_c_1, p_c_2)
    # print(p_c[0].T)
    # print(p_c.shape)
    p_c = p_c.transpose((0,2,1))[:,:,:-1]
    projected_points = projectPoints(p_c, K)
    # print(projected_points.shape)
    # projected_points_cv2, _ = cv2.projectPoints(p_t, rvec, tvec, K, None)
    # print(np.isclose(projected_points[0], projected_points_cv2.squeeze(1)))

    # print(projected_points_cv2.squeeze(1))
    # print(projected_points)
    # print(projected_points_cv2)

    # Make association between detected and projected points to compute prob and use prob to update weights
    num_points = 12
    if class_result["end-effector"] == True and class_result["base"] == True:
        points_2d = points_2d
    elif class_result["end-effector"] == True and class_result["base"] == False:
        projected_points = projected_points[:, 6:, :]
        num_points = 6
    elif class_result["end-effector"] == False and class_result["base"] == True:
        projected_points = projected_points[:, 6:, :]
        num_points = 6
    
    projected_points = projected_points.reshape(num_particles, num_points*2)
    detected_points =  points_2d.cpu().detach().numpy()
    print(detected_points.shape)
    detected_points = np.reshape(np.tile(detected_points, (num_particles, 1, 1)), (num_particles, num_points*2))
    # print(projected_points, detected_points)
    # TODO Fix NaN problem
    if np.any(np.isnan(projected_points)) or np.any(np.isnan(detected_points)):
        print(projected_points)
        print(detected_points)
    prob = rbf_kernel(projected_points, Y=detected_points).squeeze()
    return prob