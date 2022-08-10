# SPDX-FileCopyrightText: 2021 Stein Stroobants <s.stroobants@tudelft.nl>
#
# SPDX-License-Identifier: MIT
#
# Special thanks to Daniel Weber for helping with some of the implementations of the quaternion tensor functions 

import torch
import numpy as np

_pi = torch.Tensor([3.14159265358979323846])

def rad2deg(t):
    return 180. * t / _pi.to(t.device).type(t.dtype)

_conjugate_quaternion = torch.tensor([1,-1,-1,-1])

def conj_quat(q):
    return q*_conjugate_quaternion.to(q.device).type(q.dtype)

def multiply_quat(q1, q2):
    """quat1*quat2"""
    o1 = q1[..., 0] * q2[..., 0] - q1[..., 1] * q2[..., 1] - q1[..., 2] * q2[..., 2] - q1[..., 3] * q2[..., 3]
    o2 = q1[..., 0] * q2[..., 1] + q1[..., 1] * q2[..., 0] + q1[..., 2] * q2[..., 3] - q1[..., 3] * q2[..., 2]
    o3 = q1[..., 0] * q2[..., 2] - q1[..., 1] * q2[..., 3] + q1[..., 2] * q2[..., 0] + q1[..., 3] * q2[..., 1]
    o4 = q1[..., 0] * q2[..., 3] + q1[..., 1] * q2[..., 2] - q1[..., 2] * q2[..., 1] + q1[..., 3] * q2[..., 0]
    return torch.stack([o1,o2,o3,o4],dim=-1)
    

def norm_quaternion(q):
    return q / q.norm( p=2,dim=-1)[...,None]


def relative_quat(q1, q2):
    """quat1*conj(quat2)"""

    o1 =  q1[..., 0] * q2[..., 0] + q1[..., 1] * q2[..., 1] + q1[..., 2] * q2[..., 2] + q1[..., 3] * q2[..., 3]
    o2 = -q1[..., 0] * q2[..., 1] + q1[..., 1] * q2[..., 0] - q1[..., 2] * q2[..., 3] + q1[..., 3] * q2[..., 2]
    o3 = -q1[..., 0] * q2[..., 2] + q1[..., 1] * q2[..., 3] + q1[..., 2] * q2[..., 0] - q1[..., 3] * q2[..., 1]
    o4 = -q1[..., 0] * q2[..., 3] - q1[..., 1] * q2[..., 2] + q1[..., 2] * q2[..., 1] + q1[..., 3] * q2[..., 0]

    return torch.stack([o1,o2,o3,o4],dim=-1)


def diff_quat(q1,q2,norm=True):
    if norm:
        nq1 = norm_quaternion(q1)
        nq2 = norm_quaternion(q2)
    else:
        nq1 = q1
        nq2 = q2
    return relative_quat(nq1,nq2)

def safe_acos(t,eps = 4e-7):
    '''numericaly stable variant of arcuscosine'''
#     eps = 4e-8 #minimum value for acos(1) != 0
    return t.clamp(-1.0 + eps, 1.0 - eps).acos()


def relative_angle(q1,q2):
    q = diff_quat(q1,q2, norm=False)
    return 2*safe_acos(q[..., 0].abs())


def angle_loss(q1, q2):
    """
    Calculates an numerically stable version of the angle error and
    returns this combined with a tensor of zeros so it can be used in loss function
    """
    q = diff_quat(q1,q2)
    q_abs = 2 - 2*(q[..., 0]**2).sqrt()
    # q_abs = 2*(q[..., 0]**2).sqrt().arccos()
    return q_abs, torch.zeros_like(q_abs)


def relative_inclination(q1,q2):
    q = diff_quat(q1,q2, norm=False)
    return 2*(q[..., 3]**2 + q[..., 0]**2).sqrt().acos()


def inclination_loss(q1,q2):
    """
    Calculates an numerically stable version of the roll/pitch inclination angle error and
    returns this combined with a tensor of zeros so it can be used in loss function
    """
    q = diff_quat(q1,q2)
    # q_abs = 1-(q[..., 3]**2 + q[..., 0]**2).sqrt()
    q_abs = 2 - 2*(q[..., 3]**2 + q[..., 0]**2).sqrt()
    # q_abs = 2*(q[..., 3]**2 + q[..., 0]**2).sqrt().arccos()
    return q_abs, torch.zeros_like(q_abs)


def dot_loss(q1, q2):
    loss = q1[...,0] * q2[..., 0] + q1[...,1] * q2[...,1] + q1[...,2] * q2[..., 2] + q1[...,3] * q2[..., 3]
    return 1-loss, torch.zeros_like(loss)


def to_euler_angles(q):
    """
    Transform quaternion to euler angles
    expected input or of size [batchsize, sequence length, 4]
    """

    w,x,y,z= q[...,0],q[...,1],q[...,2],q[...,3]
    
    # // roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp).unsqueeze(-1)

    # // pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = sinp.arcsin().unsqueeze(-1)

    # // yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp).unsqueeze(-1)
    eulers = torch.cat((roll, pitch, yaw), 2)
    # print(f'roll size: {roll.size()}')
    # print(f'eulers size: {eulers.size()}')
    return eulers

def to_euler_angles_numpy(q):
    w, x, y, z = q[:, 0],q[:,1],q[:,2],q[:,3]

    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw]).T

    yaw = (
        -math.atan2(-2 * (q[1] * q[3] - q[0] * q[2]), q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2,)
    )

    if pitch > 0:
        pitch = pitch - np.pi
    else:
        pitch = pitch + np.pi

    eulerAngles = np.array([roll, pitch, yaw])

    return eulerAngles

# def to_euler_angles_numpy(q):
#     w, x, y, z = q[:, 0],q[:,1],q[:,2],q[:,3]

#     # roll (x-axis rotation)
#     roll = np.arcsin(2 * (y * z + w * x))
#     # pitch (y-axis rotation)
#     pitch = np.arctan2(-2 * (x * y - w * z), 
#                         w*w - x*x + y*y - z*z)


#     # yaw (z-axis rotation)
#     yaw = -np.arctan2(-2 * (x * z - w * y), w ** 2 - x ** 2 - y ** 2 + z ** 2,)
#     return np.array([roll, pitch, yaw]).T



def from_euler_angles_numpy(eulers):
    roll, pitch, yaw = eulers[:, 0], eulers[:, 1], eulers[:, 2]

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z]).T
