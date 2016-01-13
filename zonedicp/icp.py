#!/usr/bin/env python
import numpy as np
from scipy import spatial

from zonedicp import transformations


class ZonedKDTree(object):
    def __init__(self, points, zones):
        """This is an implementation of a zoned KD-tree.  It is essentially a collection
        of KD-trees, each of which indexes part of a mesh.

        The constructor expects the target points (i.e. the points to which source points will
        be aligned) and the zones defined on these points.

        points: an Nx3 matrix of mesh points.
        zones: maps zone names to the vertex indices in points."""

        self.zones = zones
        self.points = np.asarray(points)
        self._kdtrees = {}
        self._zone_indices = {}
        self._initialize()

    def _initialize(self):
        for name, point_indices in self.zones.iteritems():
            self._kdtrees[name] = spatial.cKDTree(self.points[point_indices])

    def query(self, points, regression_point_zones):
        """
        points: an Nx3 matrix of source points to match to the KDTree points
        regression_point_zones: maps zone names to indices in points
        """
        distances = np.empty((len(points),), dtype=np.float64)
        indices = np.empty((len(points),), dtype=np.int)
        for name, point_indices in regression_point_zones.iteritems():
            d, i = self._kdtrees[name].query(points[point_indices])
            distances[point_indices] = d
            indices[point_indices] = self.zones[name][i]
        return distances, indices


def _least_squares_solve(shape_model, regression_points, model_point_indices, number_of_modes):
    """
    @type shape_model: ShapeModel.ShapeModel
    """
    eigen_vectors = shape_model.eigenvectors[:number_of_modes]
    num_modes = eigen_vectors.shape[0]

    m = shape_model.mean[model_point_indices, :]
    A = eigen_vectors[:, model_point_indices, :]
    b = regression_points - m

    A = A.reshape((num_modes, -1)).T
    b = b.flatten()
    coefficients, residue, _, _ = np.linalg.lstsq(A, b)

    residue = np.sqrt(residue)
    return coefficients, residue


def fit(shape_model, transformed_regression_points, model_point_indices, sigma_threshold, number_of_modes):
    """
    @type regression_point_fitter: RegressionPointFitter.RegressionPointFitter
    """
    shape_vector, residue = _least_squares_solve(shape_model, transformed_regression_points, model_point_indices,
                                                 number_of_modes)
    singular_values = shape_model.singular_values[:len(shape_vector)]
    shape_vector = np.clip(shape_vector, -sigma_threshold * singular_values, sigma_threshold * singular_values)
    return shape_vector, residue


def compute_matrix_n(cov_matrix):
    """Compute the matrix N (refer to Horn1987, Besl1992)"""
    assert (cov_matrix.shape == (3, 3))

    trace = np.trace(cov_matrix)
    cov_matrix_t = np.transpose(cov_matrix)
    matrix_i = np.diag(np.ones(3))

    # anti-symmetric matrix A (Besl1992)
    matrix_a = cov_matrix - cov_matrix_t
    delta = np.array([matrix_a[1, 2], matrix_a[2, 0], matrix_a[0, 1]])

    # submatrix of the matrix N
    sub_matrix = cov_matrix + cov_matrix_t - trace * matrix_i

    # refer to Besl1992
    matrix_n = np.zeros((4, 4))
    matrix_n[0, 0] = trace
    matrix_n[0, 1:] = delta
    matrix_n[1:, 0] = delta
    matrix_n[1:, 1:] = sub_matrix

    return matrix_n


def quaternion_as_matrix(quaternion):
    q0, q1, q2, q3 = quaternion

    matrix_r = np.eye(4)
    matrix_r[0, 0] = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
    matrix_r[1, 1] = q0 * q0 + q2 * q2 - q1 * q1 - q3 * q3
    matrix_r[2, 2] = q0 * q0 + q3 * q3 - q1 * q1 - q2 * q2

    matrix_r[0, 1] = 2 * (q1 * q2 - q0 * q3)
    matrix_r[1, 0] = 2 * (q1 * q2 + q0 * q3)

    matrix_r[0, 2] = 2 * (q1 * q3 + q0 * q2)
    matrix_r[2, 0] = 2 * (q1 * q3 - q0 * q2)

    matrix_r[1, 2] = 2 * (q2 * q3 - q0 * q1)
    matrix_r[2, 1] = 2 * (q2 * q3 + q0 * q1)

    return matrix_r


def update_transform(source_points, target_points):
    """Compute the optimal orientation (scale, rotation, translation)
    between two point sets"""

    # substract each coordinate by centriod
    source_centroid = source_points.mean(axis=0)
    target_centroid = target_points.mean(axis=0)

    source_points = source_points - source_centroid
    target_points = target_points - target_centroid

    # compute the optimal rotation
    # compute the cross-covariance matrix
    cov_matrix = np.dot(source_points.T, target_points)
    cov_matrix /= source_points.shape[0]

    matrix_n = compute_matrix_n(cov_matrix)

    # compute the eigenvalues of the 4x4 matrix N
    eigenvalues, eigen_vectors = np.linalg.eig(matrix_n)

    # sort eigenvalues: largest -> smallest
    sorted_indices = np.argsort(eigenvalues)[::-1]
    assert (eigenvalues[sorted_indices[0]] > 0)

    # the corresponding eigen vector is selected as the optimal quaternion
    # refer Horn1987
    quaternion = eigen_vectors[:, sorted_indices[0]]

    # compute the optimal translation
    transform = quaternion_as_matrix(quaternion)
    transform[:3, 3] = target_centroid - np.dot(transform[:3, :3], source_centroid)
    return transform


def compute_error(rms_errors, distances, transform):
    rms_error = np.sqrt((distances * distances).sum()).mean()
    rms_errors.append((rms_error, transform))


def _filter_accepted_indices(points, distances, indices, accepted_indices):
    if accepted_indices is None:
        return points, distances, indices
    else:
        index_array = np.where(accepted_indices[indices])
        return points[index_array], distances[index_array], indices[index_array]


# TODO Here's the thing that actually does ICP
def icp(source_points, source_zones, target_kd_tree, max_iterations=200, tolerance=1e-5, transform=None,
        accepted_indices=None):
    original_source_points = np.array(source_points)
    target_points = target_kd_tree.points

    if transform is None:
        target_centroid = target_points.mean(axis=0)
        source_centroid = original_source_points.mean(axis=0)
        transform = transformations.compose_matrix(translate=target_centroid - source_centroid)

    source_points = transformations.apply(transform, original_source_points)

    step = 0
    rms_errors = [(0.0, transform)]

    # find closest points
    distances, indices = target_kd_tree.query(source_points, source_zones)
    source_points, distances, indices = _filter_accepted_indices(source_points, distances, indices, accepted_indices)
    # compute RMS error
    compute_error(rms_errors, distances, transform)

    while step < max_iterations and np.abs(rms_errors[-1][0] - rms_errors[-2][0]) > tolerance:
        step += 1
        # compute registration
        next_transform = update_transform(source_points, target_points[indices])
        transform = np.dot(next_transform, transform)
        source_points = transformations.apply(transform, original_source_points)
        # find closest points
        distances, indices = target_kd_tree.query(source_points, source_zones)
        source_points, distances, indices = _filter_accepted_indices(source_points, distances, indices,
                                                                     accepted_indices)
        # compute RMS error
        compute_error(rms_errors, distances, transform.copy())

    return transform, source_points, indices


def _initial_transform(side, source_points, target_points):
    """
    The means of source and target are used to apply a translation vector.
    """
    source_centroid = source_points.mean(axis=0)
    target_centroid = target_points.mean(axis=0)
    return transformations.concatenate_matrices(
            transformations.translation_matrix(target_centroid),
            transformations.compose_matrix(scale=[1.0 if side.lower() == 'right' else -1.0, 1.0, 1.0]),
            transformations.translation_matrix(-source_centroid))


def _icp_and_least_squares_fit(mesh, shape_model, ssm_zones, transform, regression_points, regression_point_zones,
                               sigma_threshold, num_modes):

    zoned_kd_tree = ZonedKDTree(mesh.points, ssm_zones)
    transform, transformed_source_points, matched_model_indices = icp(regression_points, regression_point_zones,
                                                                      zoned_kd_tree, transform=transform)
    shape_vector, fit_error = fit(shape_model, transformed_source_points, matched_model_indices, sigma_threshold,
                                  num_modes)
    mesh.points = shape_model.mean + np.tensordot(shape_model.eigenvectors[:len(shape_vector)], np.array(shape_vector),
                                                  axes=[0, 0])
    return mesh, transform, shape_vector, fit_error, matched_model_indices


def _icp_and_least_squares_fit_until_stable(mesh, shape_model, ssm_zones, transform, regression_points,
                                            regression_point_zones, sigma_threshold, num_modes):
    max_iterations = 30
    num_iterations = 1

    # In the loop we iterate with parameters new_transform, new_shape_vector, new_fit_error. Here they are initialized.
    res = _icp_and_least_squares_fit(mesh, shape_model, ssm_zones, transform, regression_points, regression_point_zones,
                                     sigma_threshold, num_modes)
    mesh, transform, shape_vector, fit_error, matched_model_indices = res
    res = _icp_and_least_squares_fit(mesh, shape_model, ssm_zones, transform, regression_points, regression_point_zones,
                                     sigma_threshold, num_modes)
    mesh, new_transform, new_shape_vector, new_fit_error, matched_model_indices = res

    singular_values = shape_model.singular_values[:len(shape_vector)]

    while num_iterations < max_iterations and (
                not np.allclose(transform, new_transform, atol=1e-4) or
                not np.allclose(new_shape_vector / singular_values, shape_vector / singular_values)):
        transform, shape_vector, fit_error = new_transform, new_shape_vector, new_fit_error
        res = _icp_and_least_squares_fit(mesh, shape_model, ssm_zones, transform, regression_points,
                                         regression_point_zones, sigma_threshold, num_modes)
        mesh, new_transform, new_shape_vector, new_fit_error, matched_model_indices = res
        num_iterations += 1

    return mesh, transform, shape_vector, new_fit_error, matched_model_indices


# FIXME This is example code from our main software that doesn't actually work without proper data
def fit(shape_model, mesh, regression_points, regression_point_zones, side, iterations=1, sigma_threshold=1.0,
        number_of_modes=5):

    transform = _initial_transform(side, regression_points, mesh.points)
    number_of_modes = min(number_of_modes, shape_model.eigenvectors.shape[0])
    ssm_zones = {zone_name: shape_model.zone_indices[zone_name] for zone_name in regression_point_zones}
    res = _icp_and_least_squares_fit_until_stable(mesh, shape_model, ssm_zones, transform, regression_points,
                                                  regression_point_zones, sigma_threshold, 1)

    mesh, transform, shape_vector, fit_error, matched_model_indices = res
    num_modes_seq = np.round(np.linspace(2, number_of_modes, iterations))

    for iteration in xrange(0, iterations):
        res = _icp_and_least_squares_fit(mesh, shape_model, ssm_zones, transform, regression_points,
                                         regression_point_zones, sigma_threshold, num_modes_seq[iteration])
        mesh, transform, shape_vector, fit_error, matched_model_indices = res
        mesh.points = shape_model.mean + np.tensordot(shape_model.eigenvectors[:len(shape_vector)],
                                                      np.array(shape_vector), axes=[0, 0])

    # We invert the transformation because the consumer of this data wants to know what transform to apply to
    # the shape model in order to get the correct result. Without inversion, the transform tells you how to go from
    # your bone to the unmodified shape model. That's NOT what you want :D
    transform = np.linalg.inv(transform)
    return shape_vector, transform
