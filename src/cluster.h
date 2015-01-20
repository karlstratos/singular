// Author: Karl Stratos (karlstratos@gmail.com)
//
// Code for clustering algorithms.

#ifndef CLUSTER_H
#define CLUSTER_H

#include <Eigen/Dense>
#include <unordered_map>

#include "util.h"

// GREEDdy agglOmerative (Greedo) clustering over n points in a Euclidean space.
// It repeatedly merges initially singleton clusters until it obtains a single
// hierarchy. Also, every merge considers at most m+1 "active" clusters where m
// is the number of leaf clusters in the hierarchy. It can be seen as a variant
// of the algorithm in: Fast and memory efficient implementation of the exact
// pnn (Franti et al., 2000).
class Greedo {
public:
    // Performs agglomerative clustering over the given *ordered* points to
    // obtain a single hierarchy with m leaf nodes. The first m points will
    // serve as the initial m active clusters, and subsequent points will be
    // added from left to right.
    void Cluster(const vector<Eigen::VectorXd> &ordered_points, size_t m);

    // Returns the mapping between bit strings and subsets in {0 ... n-1}.
    unordered_map<string, vector<size_t> > *bit2cluster() {
	return &bit2cluster_;
    }

    // Returns the average number of tightening operations performed per
    // merge because lowerbounds were not tight. This is upperbounded by m.
    double average_num_extra_tightening() {
	return ((double) num_extra_tightening_) / (num_points_ - 1);
    }

private:
    // Computes the distance between two active clusters.
    double ComputeDistance(const vector<Eigen::VectorXd> &ordered_points,
			   size_t active_index1, size_t active_index2);

    // Update two active clusters' lowerbounds / twins given their distance.
    void UpdateLowerbounds(size_t active_index1, size_t active_index2,
			   double distance);

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
    void LabelLeaves();

    // Number of data points (aka. n).
    size_t num_points_;

    // Number of clusters (aka. m).
    size_t num_clusters_;

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

    // Mapping from a bit string to a set of clustered points.
    unordered_map<string, vector<size_t> > bit2cluster_;
};

#endif  // CLUSTER_H
