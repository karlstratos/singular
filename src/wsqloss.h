// Author: Karl Stratos (karlstratos@gmail.com)
//
// Minimizer of the weighted squared (WSQ) loss between each matrix value and
// the dot product of a corresponding pair of parameter vectors.

#ifndef WSQLOSS_H
#define WSQLOSS_H

#include <Eigen/Dense>

#include "sparsesvd.h"

class WSQLossOptimizer {
public:
    // Initializes empty.
    WSQLossOptimizer() { }

    ~WSQLossOptimizer() { }

    // (Explanation of the input notation)
    //
    //    W: dxd' (sparse) matrix of weights --- missing entries assumed zero
    //    M: dxd' (sparse) matrix of corresponding values
    //    U: kxd   (dense) matrix where "U_i" := i-th column of U
    //    V: kxd'  (dense) matrix where "V_i" := i-th column of U
    //
    // Modifies the given (U, V) to minimize:
    //
    //      sum_{i=1...d, j=1...d'} W_{i,j} * (M_{i,j} - U_i' V_j)^2 +
    //               sum_{i=1...d}  regularization_term_ * ||U_i||^2 +
    //               sum_{j=1...d'} regularization_term_ * ||V_j||^2 +
    //
    // Returns the minimized loss value.
    double Optimize(SMat W, SMat M, Eigen::MatrixXd *U, Eigen::MatrixXd *V);

     // Computes the weighted squared loss from given parameters (U, V).
    double ComputeWSQLoss(SMat W, SMat M, const Eigen::MatrixXd &U,
			  const Eigen::MatrixXd &V);

    // Sets the regularization term.
    void set_regularization_term(double regularization_term) {
	regularization_term_ = regularization_term;
    }

    // Sets the learning rate prior.
    void set_learning_rate_prior(double learning_rate_prior) {
	learning_rate_prior_ = learning_rate_prior;
    }

private:
     // Gets the learning rate based on a step number.
    double GetLearningRate(size_t step);

    // Maximum number of training epochs.
    const size_t kMaxNumEpochs_ = 1000;

    // Minimum required loss reduction.
    const double kMinimumLossImprovement_ = 1e-1;

    // Regularization term.
    double regularization_term_ = 1e-7;

    // Learning rate prior.
    double learning_rate_prior_ = 1e-7;
};

#endif  // WSQLOSS_H
