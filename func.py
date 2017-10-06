__author__ = 'Shuo Yu'

import math
import numpy as np

def mag(num_list):
    sq_sum = 0
    for n in num_list:
        sq_sum += n ** 2
    return sq_sum ** 0.5


def gen_rot_mat(deg):
    rad = deg * math.pi / 180
    rot_mat = np.dot(
            np.matrix(
            [[math.cos(rad), math.sin(rad), 0],
             [-math.sin(rad), math.cos(rad), 0],
             [0, 0, 1]]
        ),
        np.matrix(
            [[1, 0, 0],
             [0, math.cos(2 * rad), math.sin(2 * rad)],
             [0, -math.sin(2 * rad), math.cos(2 * rad)]
            ]
        )
    )
    return rot_mat

def gen_rot_deg(subject_id, label_id):
    return (subject_id * 7 + label_id) * 11 * 3


def apply_rot_mat(accel_mat, deg):
    return np.dot(accel_mat, gen_rot_mat(deg))


def calc_magnitude(accel_mat, n=1):
    '''

    :param accel_mat: 2D MATRIX, a series of 3-axial accel data, [[0, 1, 0], [1, 0, 0], ...]
    :return: a vector of magnitude
    '''
    ret_vec = []
    if isinstance(accel_mat, np.matrix):
        lists = accel_mat.tolist()
    else:
        lists = accel_mat

    for row in lists:
        ret_vec.append(mag(row))

    return np.array(ret_vec)


def identify_peak(accel_mat, n=1):
    '''

    :param accel_mat: 2D MATRIX, a series of 3-axial accel data, [[0, 1, 0], [1, 0, 0], ...]
    :return: the magnitude and index of the peak in terms of magnitude
    '''
    mag_vec = calc_magnitude(accel_mat, n)
    return (np.amax(mag_vec), np.argmax(mag_vec))

def identify_valley_before_peak(accel_mat, n=1, peak_lead=25, peak_index=-1):
    '''

    :param accel_mat: 2D MATRIX, a series of 3-axial accel data, [[0, 1, 0], [1, 0, 0], ...]
    :return: the magnitude and index of the valley in terms of magnitude
    '''
    mag_vec = calc_magnitude(accel_mat, n)
    peak_index = identify_peak(accel_mat, n)[1]
    if peak_index == 0:
        return mag_vec[peak_index], 0
    else:
        return (np.amin(mag_vec[max(peak_index - peak_lead, 0):peak_index]),
                np.argmin(mag_vec[max(peak_index - peak_lead, 0):peak_index]))


def sample_around_peak(accel_mat, before=24, after=32):
    '''
    Generate a snippet of acceleration sample around the peak detected
    :param accel_mat:
    :param before:
    :param after:
    :return:
    '''
    peak_index = identify_peak(accel_mat)[1]
    # print('peak_index: {}'.format(peak_index))
    start = peak_index - before if peak_index - before >= 0 else 0
    end = peak_index + after if peak_index + after <= accel_mat.shape[0] else accel_mat.shape[0]
    temp_mat = accel_mat[start:end, :]
    if peak_index - before < 0:
        pre = np.array(accel_mat[0, :].tolist() * (before - peak_index)).reshape(-1, 3)
        temp_mat = np.concatenate((np.matrix(pre), temp_mat))
    if peak_index + after > accel_mat.shape[0]:
        post = np.array(accel_mat[-1, :].tolist() * (peak_index + after - accel_mat.shape[0])).reshape(-1, 3)
        # print(post)
        temp_mat = np.concatenate((temp_mat, np.matrix(post)))

    return temp_mat


def convert_to_magnitude_list(mat_list):
    ret_list = []
    for mat in mat_list:
        ret_list.append(calc_magnitude(mat))
    return ret_list


def calc_gravity(accel_mat):
    """
    Use the acceleration vectors (1-sec period) after the shock to estimate an axis
    :param accel_mat:
    :return:
    """
    # p = identify_peak(accel_mat)[1]
    r, c = accel_mat.shape
    p = 0
    # print(p)
    if r > 48:
        return np.mean(accel_mat[p+36:p+48, :], axis=0)
    else:
        return np.mean(accel_mat[p:p+12, :], axis=0)


def calc_gravity_o(accel_mat):
    """
    Use the acceleration vectors (1-sec period) after the shock to estimate an axis
    :param accel_mat:
    :return:
    """
    # p = identify_peak(accel_mat)[1]
    r, c = accel_mat.shape
    p = 0
    return np.mean(accel_mat[p:p+12, :], axis=0)


def calc_vt_comp(accel_mat):
    vt_vec = calc_gravity(accel_mat)
    unit_vt_vec = vt_vec / np.linalg.norm(vt_vec)
    # a_gi = (a_i * g^T) * g
    # ret_mat = np.dot(np.dot(accel_mat, np.transpose(unit_vt_vec)), unit_vt_vec) - vt_vec
    ret_mat = np.dot(np.dot(accel_mat, np.transpose(unit_vt_vec)), unit_vt_vec)
    return ret_mat


def calc_rem_comp_excl_vt(accel_mat):
    ret_mat = accel_mat - calc_vt_comp(accel_mat)
    return ret_mat


def directed_vec_mag(accel_mat, vec):
    ret_list = []
    for i in range(accel_mat.shape[0]):
        a = accel_mat[i, 0] * 1000 / vec[0, 0]
        ret_list.append(a)
    return np.matrix(np.array(ret_list).reshape(-1, 1))


def calc_vt_comp_with_rem_mag(accel_mat):
    vt_vec = calc_gravity(accel_mat)
    # unit_vt_vec = vt_vec / np.linalg.norm(vt_vec)
    vt_comp_mat = calc_vt_comp(accel_mat)
    # print('vt_comp_mat:', vt_comp_mat)
    vt_mag_mat = directed_vec_mag(vt_comp_mat, vt_vec)
    rem_mag_mat = np.matrix(calc_magnitude((accel_mat - vt_comp_mat) * 1000 / np.linalg.norm(vt_vec)).reshape(-1, 1))
    return np.concatenate((vt_mag_mat, rem_mag_mat), axis=1)


def calibrate(accel_mat, calib=1):
    ret_mat = None
    if calib == 1:
        ret_mat = calc_vt_comp_with_rem_mag(accel_mat)
    elif calib == 2:
        ret_mat = np.matrix(calc_magnitude(accel_mat).reshape(-1, 1))
    else:
        ret_mat = accel_mat
    return ret_mat


def mat_to_g(value_list):
    """

    :param value_list: n x 3 matrix or list
    :return: n x 1 matrix
    """
    if not isinstance(value_list, np.matrix):
        mat = np.matrix(value_list)
    else:
        mat = value_list
    mat = np.linalg.norm(mat, ord=2, axis=1, keepdims=True)
    return mat


def mat_to_vc(value_list):
    """

    :param value_list: n x 3 matrix or list
    :return: n x 2 matrix
    """
    if not isinstance(value_list, np.matrix):
        mat = np.matrix(value_list)
    else:
        mat = value_list
    mat = calc_vt_comp_with_rem_mag(mat)
    return mat


def mat_to_peak(value_list):
    """

    :param value_list: n x 3 matrix or list
    :return: n x 1 matrix
    """
    if not isinstance(value_list, np.matrix):
        mat = np.matrix(value_list)
    else:
        mat = value_list
    return identify_peak(mat)[0]