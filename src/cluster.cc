// Author: Karl Stratos (karlstratos@gmail.com)

#include "cluster.h"

#include <limits>

bool KMeansSolver::Cluster(const vector<Eigen::VectorXd> &ordered_points,
			   size_t K, vector<size_t> *cluster_mapping,
			   size_t max_num_iterations) {
    ASSERT(ordered_points.size() > 0, "Empty set of points.");
    size_t dimension = ordered_points[0].size();

    ASSERT(K <= ordered_points.size(), "Number of clusters " << K << " bigger "
	   << "than number of points " << ordered_points.size());

    // Initialize the K centroids as the first K of the given points.
    vector<Eigen::VectorXd> centroid(K);
    for (size_t cluster_num = 0; cluster_num < K; ++cluster_num) {
	centroid[cluster_num] = ordered_points[cluster_num];
    }

    // Repeatedly update the cluster mapping by alternating between Step 1 and
    // Step 2 until convergence.
    cluster_mapping->clear();
    cluster_mapping->resize(ordered_points.size());
    size_t num_iterations = 0;
    bool clustering_is_converged = false;
    while (num_iterations < max_num_iterations) {
	// Step 1: Assign each point to the closest centroid (cluster).
	bool clusters_have_not_changed = true;
	for (size_t point_num = 0; point_num < ordered_points.size();
	     ++point_num) {
	    double min_squared_distance = numeric_limits<double>::infinity();
	    size_t closest_centroid_num = 0;
	    for (size_t cluster_num = 0; cluster_num < K; ++cluster_num) {
		Eigen::VectorXd diff =
		    ordered_points[point_num] - centroid[cluster_num];
		double squared_distance = diff.squaredNorm();
		if (squared_distance < min_squared_distance) {
		    closest_centroid_num = cluster_num;
		    min_squared_distance = squared_distance;
		}
	    }
	    if (cluster_mapping->at(point_num) != closest_centroid_num) {
		clusters_have_not_changed = false;
	    }
	    (*cluster_mapping)[point_num] = closest_centroid_num;
	}
	if (clusters_have_not_changed) {
	    clustering_is_converged = true;
	    break;
	}

	// Step 2: Re-compute centroids as the means of clustered points.
	centroid.clear();
	centroid.resize(K);
	vector<size_t> cluster_size(K);  // How many points in a cluster?
	for (size_t point_num = 0; point_num < ordered_points.size();
	     ++point_num) {
	    size_t assigned_cluster_num = cluster_mapping->at(point_num);
	    if (centroid[assigned_cluster_num].size() == 0) {
		centroid[assigned_cluster_num] = ordered_points[point_num];
	    } else {
		centroid[assigned_cluster_num] += ordered_points[point_num];
	    }
	    ++cluster_size[assigned_cluster_num];
	}
	for (size_t cluster_num = 0; cluster_num < K; ++cluster_num) {
	    centroid[cluster_num] /= cluster_size[cluster_num];

	    if (centroid[cluster_num].size() == 0) {
		// For an empty cluster, set the centroid in the "outer space".
		centroid[cluster_num].resize(dimension);
		for (size_t i = 0; i < dimension; ++i) {
		    centroid[cluster_num](i) =
			numeric_limits<double>::infinity();
		}
	    }
	}
	++num_iterations;
    }
    return clustering_is_converged;
}
