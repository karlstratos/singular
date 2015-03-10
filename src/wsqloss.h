// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Optimizer of the weighted squared (WSQ) loss between each matrix value and
// the dot product of a corresponding pair of parameter vectors.

#ifndef WSQLOSS_H
#define WSQLOSS_H

#include <Eigen/Dense>
#include <iostream>
#include <unordered_map>
#include <vector>

using namespace std;

typedef unordered_map<size_t, vector<tuple<size_t, double, double> > > WSQMap;

class WSQLossOptimizer {
public:
    // Initializes empty.
    WSQLossOptimizer() { }

    ~WSQLossOptimizer() { }

    //
    void Optimize(const WSQMap &col2row, const WSQMap &row2col,
		  Eigen::MatrixXd *U, Eigen::MatrixXd *V);


    double ComputeLoss(const WSQMap &col2row, const Eigen::MatrixXd &U,
		       const Eigen::MatrixXd &V);

private:
    // Minimum required loss reduction.
    const double kMinimumLossImprovement_ = 1e-1;

    // Maximum number of training epochs.
    size_t max_num_epochs_ = 100;
};

#endif  // WSQLOSS_H
