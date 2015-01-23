// Author: Karl Stratos (karlstratos@gmail.com)

#include "cluster.h"

#include <cfloat>
#include <limits>
#include <stack>

void Greedo::Cluster(const vector<Eigen::VectorXd> &ordered_points, size_t m) {
    size_t n = ordered_points.size();  // n = number of points.
    ASSERT(m <= n, "Number of clusters " << m << " is smaller than number of "
	   << "points: " << n);
    num_points_ = n;
    num_clusters_ = m;

    //--------------------------------------------------------------------------
    // (Sketch of the algorithm)
    // We compute an ordered list of 2n-1 clusters (n original points as
    // singletons + n-1 merges):
    //    [Singletons]       0     1     2   ...    n-2   n-1
    //    [Non-singletons]   n   n+1   n+2   ...   2n-2
    // Normally in agglomerative clustering, we would start with the first n
    // singleton clusters and repeatedly merge: with the Franti et al. (2000)
    // trick, this would take O(gdn^2) where g is a data-dependent constant
    // and d is the dimension of the vector space.
    //
    // Here, we will instead start with the first m singleton clusters and
    // repeatedly merge. At every iteration, we will include the next singleton
    // cluster for consideration. Thus in any given moment, we handle at most
    // m+1 "active" clusters. Again applying the Franti et al. (2000) trick,
    // this now taks O(gdmn).
    //--------------------------------------------------------------------------

    Z_.resize(n - 1);  // Information about the n-1 merges.
    size_.resize(2 * n - 1);  // Clusters' sizes.
    active_.resize(m + 1);  // Active clusters.
    mean_.resize(m + 1);  // Clusters' means.
    lb_.resize(m + 1);  // Lowerbounds.
    twin_.resize(m + 1);  // Indices in {0 ... m} for merge candidates.
    tight_.resize(m + 1);  // Is the current lowerbound tight?
    num_extra_tightening_ = 0;  // Number of tightening operations.

    // Initialize the first m clusters.
    for (size_t a1 = 0; a1 < m; ++a1) {  // Tightening m clusters: O(dm^2).
	size_[a1] = 1;
	active_[a1] = a1;
	mean_[a1] = ordered_points[a1];
	lb_[a1] = DBL_MAX;
	for (size_t a2 = 0; a2 < a1; ++a2) {
	    double dist = ComputeDistance(ordered_points, a1, a2);
	    UpdateLowerbounds(a1, a2, dist);
	}
    }

    // Main loop: Perform n-1 merges.
    size_t next_singleton = m;
    for (size_t merge_num = 0; merge_num < n - 1; ++merge_num) {
	if (next_singleton < n) {
	    // Set the next remaining point as the (m+1)-th active cluster.
	    size_[next_singleton] = 1;
	    active_[m] = next_singleton;
	    mean_[m] = ordered_points[next_singleton];
	    lb_[m] = DBL_MAX;
	    for (int a = 0; a < m; ++a) {  // Tightening 1 cluster: O(dm).
		double dist = ComputeDistance(ordered_points, m, a);
		UpdateLowerbounds(m, a, dist);
	    }
	    ++next_singleton;
	}

	// Number of active clusters is m+1 until all singleton clusters are
	// active. Then it decreases m ... 2.
	size_t num_active_clusters = min(m + 1, n - merge_num);

	// Find which active cluster has the smallest lowerbound: O(m).
	size_t candidate_index = 0;
	double smallest_lowerbound = DBL_MAX;
	for (size_t a = 0; a < num_active_clusters; ++a) {
	    if (lb_[a] < smallest_lowerbound) {
		smallest_lowerbound = lb_[a];
		candidate_index = a;
	    }
	}

	while (!tight_[candidate_index]) {
	    // The current candidate turns out to have a loose lowerbound.
	    // Tighten it: O(dm).
	    lb_[candidate_index] = DBL_MAX;  // Recompute lowerbound.
	    for (size_t a = 0; a < num_active_clusters; ++a) {
		if (a == candidate_index) continue;  // Skip self.
		double dist =
		    ComputeDistance(ordered_points, candidate_index, a);
		UpdateLowerbounds(candidate_index, a, dist);
	    }
	    ++num_extra_tightening_;

	    // Again, find an active cluster with the smallest lowerbound: O(m).
	    smallest_lowerbound = DBL_MAX;
	    for (size_t a = 0; a < num_active_clusters; ++a) {
		if (lb_[a] < smallest_lowerbound) {
		    smallest_lowerbound = lb_[a];
		    candidate_index = a;
		}
	    }
	}

	// At this point, we have a pair of active clusters with minimum
	// pairwise distance. Denote their active indices by "alpha" and "beta".
	size_t alpha = candidate_index;
	size_t beta = twin_[alpha];
	if (alpha > beta) {  // WLOG, we will maintain alpha < beta.
	    size_t temp = alpha;
	    alpha = beta;
	    beta = temp;
	}

	// Cluster whose twin was in {alpha, beta} has a loose lowerbound.
	for (size_t a = 0; a < num_active_clusters; ++a) {
	    if (twin_[a] == alpha || twin_[a] == beta) { tight_[a] = false; }
	}

	// Record the merge in Z_.
	size_t merged_cluster = n + merge_num;
	get<0>(Z_[merge_num]) = active_[alpha];
	get<1>(Z_[merge_num]) = active_[beta];
	get<2>(Z_[merge_num]) = smallest_lowerbound;

	// Compute the size of the merged cluster.
	size_[merged_cluster] = size_[active_[alpha]] + size_[active_[beta]];

	// MUST compute the merge mean before modifying active clusters!
	ComputeMergedMean(ordered_points, alpha, beta, &mean_[alpha]);

	//----------------------------------------------------------------------
	// SHIFTING (Recall: alpha < beta)
	// We now replace the (alpha)-th active cluster with the new merged
	// cluster. Then we will shift active clusters past index beta to the
	// left by one position to overwrite beta. Graphically speaking, the
	// current M <= m+1 active clusters will change in structure (1 element
	// shorter) as follows:
	//
	//     a_1   ...   alpha        ...  a   b   beta   c   d   ...   a_M
	// =>
	//     a_1   ...   alpha+beta   ...  a   b   c   d   ...   a_M
	//----------------------------------------------------------------------

	// Set the merged cluster as the (alpha)-th active cluster and tighten.
	active_[alpha] = merged_cluster;
	lb_[alpha] = DBL_MAX;
	for (size_t a = 0; a < num_active_clusters; ++a) {
	    if (a == alpha) continue;  // Skip self.
	    if (a == beta) continue;  // beta will be overwritten anyway.
	    double dist = ComputeDistance(ordered_points, alpha, a);
	    UpdateLowerbounds(alpha, a, dist);
	}

	// Shift the elements past beta to the left by one (overwriting beta).
	for (size_t a = 0; a < num_active_clusters - 1; ++a) {
	    if (a < beta && twin_[a] > beta) {
		// Even for non-shifting elements, if their twin index is
		// greater than beta, we must shift accordingly.
		twin_[a] = twin_[a] - 1;
	    }

	    if (a >= beta) {
		active_[a] = active_[a + 1];
		mean_[a] = mean_[a + 1];
		lb_[a] = lb_[a + 1];
		tight_[a] = tight_[a + 1];

		if (twin_[a + 1] < beta) {
		    twin_[a] = twin_[a + 1];
		} else {  // Again, need to shift twin indices accordingly.
		    twin_[a] = twin_[a + 1] - 1;
		}
	    }
	    ASSERT(a != twin_[a], "Active index " << a << " has itself for "
		   "twin: something got screwed while shifting");
	}
    }

    // Organize merges so that the right child cluster is always more recent
    // than the left child cluster.
    for (size_t i = 0; i < n - 1; ++i) {
	if (get<0>(Z_[i]) > get<1>(Z_[i])) {
	    double temp = get<0>(Z_[i]);
	    get<0>(Z_[i]) = get<1>(Z_[i]);
	    get<1>(Z_[i]) = temp;
	}
    }
    LabelLeaves();  // Clustering done: label bit strings.
}

double Greedo::ComputeDistance(const vector<Eigen::VectorXd> &ordered_points,
			       size_t active_index1, size_t active_index2) {
    size_t size1 = size_[active_[active_index1]];
    size_t size2 = size_[active_[active_index2]];
    double scale = 2.0 * size1 * size2 / (size1 + size2);
    Eigen::VectorXd diff = mean_[active_index1] - mean_[active_index2];
    return scale * diff.squaredNorm();
}

void Greedo::UpdateLowerbounds(size_t active_index1, size_t active_index2,
			       double distance) {
    if (distance < lb_[active_index1]) {
	lb_[active_index1] = distance;
	twin_[active_index1] = active_index2;
	tight_[active_index1] = true;
    }
    if (distance < lb_[active_index2]) {
	lb_[active_index2] = distance;
	twin_[active_index2] = active_index1;
	tight_[active_index2] = true;
    }
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

    size_t n = num_points_;
    size_t m = num_clusters_;

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
