// Author: Karl Stratos (karlstratos@gmail.com)

#include "cluster.h"

#include <cfloat>
#include <limits>

void Greedo::Cluster(const vector<Eigen::VectorXd> &ordered_points, size_t m) {
    size_t n = ordered_points.size();  // n = number of points.
    ASSERT(m <= n, "Number of clusters " << m << " is smaller than number of "
	   << "points: " << n);
    size_t d = ordered_points[0].size();  // d = dimension of the vector space

    Z_.resize(n - 1);  // Information about the n-1 merges.
    size_.resize(2 * n - 1);  // Clusters' sizes.
    active_.resize(m + 1);  // Active clusters.
    mean_.resize(m + 1);  // Clusters' means.
    lb_.resize(m + 1);  // Lowerbounds.
    twin_.resize(m + 1);  // Indices in {0 ... m} for merge candidates.
    tight_.resize(m + 1);  // Is the current lowerbound tight?
    num_extra_tightening_ = 0;  // Reset the number of tightening operations.

    // Initialize the first m clusters (also active), and tighten them.
    // This step involves O(dm^2) computations.
    for (size_t i = 0; i < m; ++i) {
	size_[i] = 1;
	active_[i] = i;
	mean_[i] = ordered_points[i];
	lb_[i] = DBL_MAX;

	for (int j = 0; j < i; ++j) {
	    double dist = ComputeDistance(ordered_points, i, j);
	    if (dist < lb_[i]) {  // Tighten the i-th active cluster.
		lb_[i] = dist;
		twin_[i] = j;
		tight_[i] = true;
	    }
	    if (dist < lb_[j]) {  // Tighten the j-th active cluster.
		lb_[j] = dist;
		twin_[j] = i;
		tight_[j] = true;
	    }
	}
    }

    // Main loop: Perform n-1 merges.
    size_t next_singleton = m;
    for (size_t merge_num = 0; merge_num < n - 1; ++merge_num) {
	if (next_singleton < n) {
	    // Set the next ordered point as the (m+1)-th active (singleton)
	    // cluster and tighten it.
	    size_[next_singleton] = 1;
	    active_[m] = next_singleton;
	    mean_[m] = ordered_points[next_singleton];
	    lb_[m] = DBL_MAX;

	    for (int j = 0; j < m; ++j) {
		double dist = ComputeDistance(ordered_points, m, j);
		if (dist < lb_[m]) {  // Tighten the m-th active cluster.
		    lb_[m] = dist;
		    twin_[m] = j;
		    tight_[m] = true;
		}
		if (dist < lb_[j]) {  // Tighten the j-th active cluster.
		    lb_[j] = dist;
		    twin_[j] = m;
		    tight_[j] = true;
		}
	    }
	    ++next_singleton;
	}

	// Number of active clusters is m+1 until all singleton clusters are
	// active. Then it decreases as: m ... 2.
	size_t num_active_clusters = min(m + 1, n - merge_num);

	// Find which active cluster has the smallest lowerbound. This requires
	// just O(m) computations.
	size_t candidate_index = 0;
	double smallest_lowerbound = DBL_MAX;
	for (size_t i = 0; i < num_active_clusters; ++i) {
	    if (lb_[i] < smallest_lowerbound) {
		smallest_lowerbound = lb_[i];
		candidate_index = i;
	    }
	}

	while (!tight_[candidate_index]) {
	    // Tighten this candidate, which requires O(dm) computations.
	    lb_[candidate_index] = DBL_MAX;  // Recompute lowerbound.
	    for (size_t i = 0; i < num_active_clusters; ++i) {
		if (i == candidate_index) continue;  // Skip self.
		double dist =
		    ComputeDistance(ordered_points, candidate_index, i);
		if (dist < lb_[candidate_index]) {  // Tighten candidate.
		    lb_[candidate_index] = dist;
		    twin_[candidate_index] = i;
		    tight_[candidate_index] = true;
		}
		if (dist < lb_[i]) {  // Tighten the i-th active cluster.
		    lb_[i] = dist;
		    twin_[i] = candidate_index;
		    tight_[i] = true;
		}
	    }
	    num_extra_tightening_++;

	    // Again, find which active cluster has the smallest lowerbound in
	    // O(m) computations.
	    smallest_lowerbound = DBL_MAX;
	    for (size_t i = 0; i < num_active_clusters; ++i) {
		if (lb_[i] < smallest_lowerbound) {
		    smallest_lowerbound = lb_[i];
		    candidate_index = i;
		}
	    }
	}

	// At this point, we have a pair of active clusters with minimum
	// distance. Denote their active indices by "a" and "b".
	size_t a = candidate_index;
	size_t b = twin_[a];
	if (a > b) {  // WLOG, we will maintain a < b.
	    size_t temp = a;
	    a = b;
	    b = temp;
	}

	// Every cluster whose twin was in {a, b} now has a loose lowerbound.
	for (size_t i = 0; i < num_active_clusters; ++i) {
	    if (twin_[i] == a || twin_[i] == b) { tight_[i] = false; }
	}

	// Record the merge in Z_.
	size_t merged_cluster = n + merge_num;
	get<0>(Z_[merge_num]) = active_[a];
	get<1>(Z_[merge_num]) = active_[b];
	get<2>(Z_[merge_num]) = smallest_lowerbound;
	size_[merged_cluster] = size_[active_[a]] + size_[active_[b]];

	// We now need to replace the a-th active cluster with this new merged
	// cluster. Then we will shift active clusters past index b to the left
	// by one position to overwrite b.

	// MUST compute the merge before changing active clusters!
	ComputeMergedMean(ordered_points, a, b, &mean_[a]);
	active_[a] = merged_cluster;
	lb_[a] = DBL_MAX;
	for (size_t i = 0; i < num_active_clusters; ++i) {
	    if (i == a) continue;  // Skip self.
	    if (i == b) continue;  // Active b will be overwritten anyway.
	    double dist = ComputeDistance(ordered_points, a, i);
	    if (dist < lb_[a]) {  // Tighten the a-th active cluster.
		lb_[a] = dist;
		twin_[a] = i;
		tight_[a] = true;
	    }
	    if (dist < lb_[i]) {  // Tighten the i-th active cluster.
		lb_[i] = dist;
		twin_[i] = a;
		tight_[i] = true;
	    }
	}

	// Now shift the elements past b to the left by one (overwriting b).
	for (size_t i = 0; i < num_active_clusters - 1; ++i) {
	    if (i < b && twin_[i] > b) {
		// Even for non-shifting elements, if their twin index is
		// greater than b, we must shift accordingly.
		twin_[i] = twin_[i] - 1;
	    }

	    if (i >= b) {
		active_[i] = active_[i + 1];
		mean_[i] = mean_[i + 1];
		lb_[i] = lb_[i + 1];

		// Again, need to shift twin indices accordingly.
		if (twin_[i + 1] < b) {
		    twin_[i] = twin_[i + 1];
		} else {
		    twin_[i] = twin_[i + 1] - 1;
		}
		tight_[i] = tight_[i + 1];
	    }

	    ASSERT(i != twin_[i], "Active index " << i << " has itself for "
		   "twin: something got screwed while shifting");
	}
    }
}

double Greedo::ComputeDistance(const vector<Eigen::VectorXd> &ordered_points,
			       size_t active_index1, size_t active_index2) {
    size_t size1 = size_[active_[active_index1]];
    size_t size2 = size_[active_[active_index2]];
    double scale = 2.0 * size1 * size2 / (size1 + size2);
    Eigen::VectorXd diff = mean_[active_index1] - mean_[active_index2];
    return scale * diff.squaredNorm();
}

void Greedo::ComputeMergedMean(const vector<Eigen::VectorXd> &ordered_points,
			       size_t active_index1, size_t active_index2,
			       Eigen::VectorXd *new_mean) {
    double size1 = size_[active_[active_index1]];
    double size2 = size_[active_[active_index2]];
    double total_size = size1 + size2;
    double scale1 = size1 / total_size;
    double scale2 = size2 / total_size;
    *new_mean = scale1 * mean_[active_index1] + scale2 * mean_[active_index2];
}

void Greedo::LabelLeaves(unordered_map<string, vector<size_t> >& bit2cluster) {
    ASSERT(Z_.size() > 0, "No merge information to label leaves!");

    // Recover the number of points, n, from the size of Z_, n-1.
    size_t n = Z_.size() + 1;

    // Use breadth-first search (BFS) to traverse the tree. Maintain bit strings
    // to mark the path from the root.
    stack<pair<size_t, string> > bfs_stack;  // [  ... (77, "10011") ]

    // Push the root cluster (2n-2) with an empty bit string.
    bfs_stack.push(make_pair(2 * n - 2, ""));

    while(!bfs_stack.empty()){
        std::pair<size_t,string> cluster_bits_pair = bfs_stack.top();
        bfs_stack.pop();
        size_t cluster = cluster_bits_pair.first;
        string bitstring = cluster_bits_pair.second;
        // if node < n, it's a leaf node
        if (node < n) {
	    subtree[bits].push_back(node);
	}
        else
        {
            size_t node1 = Z[node-n][0];
            size_t node2 = Z[node-n][1];

            string left_bits = bits;
            string right_bits = bits;

            if (node >= 2*n - m)
            {
                left_bits = left_bits + "0";
                right_bits = right_bits + "1";
            }

            bfs_stack.push(make_pair(node1,left_bits));
            bfs_stack.push(make_pair(node2,right_bits));
        }
    }
}

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
