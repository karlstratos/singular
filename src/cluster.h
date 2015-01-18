// Author: Karl Stratos (karlstratos@gmail.com)
//
// Code for clustering algorithms.

#ifndef CLUSTER_H
#define CLUSTER_H

#include <Eigen/Dense>
#include <unordered_map>

#include "util.h"

// GREEDdy agglOmerative (Greedo) clustering over n points in a Euclidean space.
// It repeatedly merges a pair of clusters (initially singletons) to obtain a
// single hierarchy. For efficiency, every merge considers at most m+1 "active"
// clusters (initially m + 1 singletons). The resulting hierarchy has m leaf
// clusters.
//
// Clustering involves computing an ordered list of 2n-1 clusters:
//    (n original points)   0     1     2   ...    n-2   n-1
//    (n-1 merges)          n   n+1   n+2   ...   2n-2
// Greedo recovers a final hierarchy by keeping track of which clusters merge
// together and by pruning below the m clusters closest to the root.
class Greedo {
public:
    // Perform agglomerative clustering over the given *ordered* points to
    // obtain a single hierarchy with m leaf nodes. The first m points will
    // serve as initial m clusters in the algorithm.
    void Cluster(const vector<Eigen::VectorXd> &ordered_points, size_t m);
private:
    // Computes the distance between two active clusters.
    double ComputeDistance(const vector<Eigen::VectorXd> &ordered_points,
			   size_t active_index1, size_t active_index2);

    // Computes the new mean resulting from merging two active clusters.
    void ComputeMergedMean(const vector<Eigen::VectorXd> &ordered_points,
			   size_t active_index1, size_t active_index2,
			   Eigen::VectorXd *new_mean);

    // Based on the computed hierarchy, create a mapping from a leaf-node bit
    // string indicating the path from the root to the associated clusters.
    //                    ...
    //                   /  \
    //                1010  1011
    //             {0,3,9}   {77,1,8}
    void LabelLeaves(unordered_map<string, vector<size_t> >& bit2cluster);

    // Information of the n-1 merges. For i in {0 ... n-2}:
    //    get<0>(Z_[i]) = left child of cluster n+i
    //    get<1>(Z_[i]) = right child of cluster n+i
    //    get<2>(Z_[i]) = distance between children for cluster n+i
    vector<tuple<size_t, size_t, double> > Z_;

    // For c = 0 ... 2n-2:
    //    size_[c] = number of elements in cluster c.
    vector<size_t> size_;

    // For i = 0 ... m:
    //    active_[i] = i-th active cluster, an element in {0 ... 2n-2}.
    vector<size_t> active_;

    // For i = 0 ... m:
    //    mean_[i] = mean of the i-th active cluster.
    vector<Eigen::VectorXd> mean_;

    // For i = 0 ... m:
    //    lb_[i] = lowerbound on the distance from the i-th active cluster to
    //             any other active cluster.
    vector<double> lb_;

    // For i = 0 ... m:
    //    twin_[i] = index in {0 ... m}\{i} that indicates which active cluster
    //               is estimated as the nearest to the i-th active cluster.
    vector<size_t> twin_;

    // For i = 0 ... m:
    //    tight_[i] = true if lb_[i] is tight.
    vector<bool> tight_;

    // Total number of tightening operations performed because lowerbounds were
    // not tight.
    size_t num_extra_tightening_ = 0;
};

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
