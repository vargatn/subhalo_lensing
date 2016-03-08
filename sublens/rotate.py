import numpy as np


def rotmatr(angle, axis_vector, deg=True):
    """
    3D rotation matrix around specified axis_vector

    :param angle: angle
    :param axis_vector: normal vector
    :param deg: if True use degrees, else radians
    :return: Rotation matrix
    """

    if deg:
        angle *= np.pi / 180.

    u = axis_vector / np.sqrt(np.sum(axis_vector ** 2.))
    ux = np.array([[0., -u[2], u[1]], [u[2], 0., -u[0]], [-u[1], u[0], 0]])
    udot = np.outer(u, u)
    R = np.eye(3) * np.cos(angle)  + np.sin(angle) * ux +\
        (1. - np.cos(angle)) * udot

    return np.matrix(R)


def spher2cart(spher, deg=True):
    """Spherical to cartesian"""
    spher = spher.copy()
    if deg:
        spher *= np.pi / 180
    x = np.cos(spher[0]) * np.cos(spher[1])
    y = np.sin(spher[0]) * np.cos(spher[1])
    z = np.sin(spher[1])

    cart = np.array([x, y, z])
    return cart


def cart2spher(cart, deg=True):
    """Cartesian to Spherical"""
    cart = cart.copy()
    r = np.sqrt(np.sum(cart ** 2.))
    az = np.arctan2(cart[1], cart[0])
    lat = np.arcsin(cart[2] / r)

    spher = np.array([az, lat])
    if deg:
        spher  *= 180. / np.pi
    return spher


def rot_center(rd, cent, angle=180, deg=True):
    """
    Rotates the rd point around the axis defined by the center point cent

    :param rd: point pair (RA, DEC)
    :param cent:rotation center, default is (0, 0)
    :return: rotated position (RA, DEC)
    """

    rax = spher2cart(cent)
    rax_angle = -angle
    Rot = rotmatr(rax_angle, rax, deg=deg)

    cart = spher2cart(rd)
    cart = np.expand_dims(cart, axis=1)

    res_rot = np.array(np.dot(Rot, np.matrix(cart))).transpose((1, 0)).T
    new_rd = cart2spher(res_rot[:, 0])
    return new_rd