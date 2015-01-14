// Author: Karl Stratos (karlstratos@gmail.com)
//
// Code for clustering algorithms.

#ifndef CLUSTER_H
#define CLUSTER_H

#include <Eigen/Dense>
#include <unordered_map>

#include "util.h"

class KMeansSolver {
public:
    // Clusters given points into K groups using K-means. It assumes that
    // points are ordered such that the first K serve as initial centroids.
    // The value cluster_mapping[i] in [0, K) indicates the cluster of the
    // i-th point. It returns true if clustering has converged within the given
    // number of iterations, false otherwise.
    bool Cluster(const vector<Eigen::VectorXd> &ordered_points,
		 size_t K, vector<size_t> *cluster_mapping,
		 size_t max_num_iterations);
};

#endif  // CLUSTER_H
