// Author: Karl Stratos (karlstratos@gmail.com)

#include "cluster.h"

#include <cfloat>
#include <limits>
#include <stack>

void Greedo::Cluster(const vector<Eigen::VectorXd> &ordered_points, size_t m) {
    size_t n = ordered_points.size();  // n = number of points.
    ASSERT(m <= n, "Number of clusters " << m << " is smaller than number of "
	   << "points: " << n);

    // Note that we will compute an ordered list of 2n-1 clusters:
    //    (n original points)   0     1     2   ...    n-2   n-1
    //           (n-1 merges)   n   n+1   n+2   ...   2n-2
    Z_.resize(n - 1);  // Information about the n-1 merges.
    size_.resize(2 * n - 1);  // Clusters' sizes.
    active_.resize(m + 1);  // Active clusters.
    mean_.resize(m + 1);  // Clusters' means.
    lb_.resize(m + 1);  // Lowerbounds.
    twin_.resize(m + 1);  // Indices in {0 ... m} for merge candidates.
    tight_.resize(m + 1);  // Is the current lowerbound tight?
    num_extra_tightening_ = 0;  // Number of tightening operations.

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
    LabelLeaves();
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

void Greedo::LabelLeaves() {
    ASSERT(Z_.size() > 0, "No merge information to label leaves!");
    ASSERT(active_.size() > 0, "Active clusters missing!");
    bit2cluster_.clear();

    // Recover the number of points, n, from the size of Z_, n-1.
    size_t n = Z_.size() + 1;

    // Recover the number of clusters, m, from the size of active_, m+1.
    size_t m = active_.size() - 1;

    // Use breadth-first search (BFS) to traverse the tree. Maintain bit strings
    // to mark the path from the root.
    stack<pair<size_t, string> > bfs_stack;  // [  ... (77, "10011") ]

    // Push the root cluster (2n-2) with an empty bit string.
    bfs_stack.push(make_pair(2 * n - 2, ""));

    while(!bfs_stack.empty()){
        std::pair<size_t,string> cluster_bitstring_pair = bfs_stack.top();
        bfs_stack.pop();
        size_t cluster = cluster_bitstring_pair.first;
        string bitstring = cluster_bitstring_pair.second;

        if (cluster < n) {
	    // We have a leaf cluster. Add to the current bit string.
	    bit2cluster_[bitstring].push_back(cluster);
	} else {
	    // We have a non-leaf cluster. Branch to its two children.
            size_t left_child_cluster = get<0>(Z_[cluster - n]);
            size_t right_child_cluster = get<1>(Z_[cluster - n]);

            string left_bitstring = bitstring;
            string right_bitstring = bitstring;
            if (cluster >= 2 * n - m) {
		// Prune branches to have only m leaf clusters.
                left_bitstring += "0";
                right_bitstring += "1";
            }

            bfs_stack.push(make_pair(left_child_cluster, left_bitstring));
            bfs_stack.push(make_pair(right_child_cluster, right_bitstring));
        }
    }
}
