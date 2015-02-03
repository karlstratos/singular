// Author: Karl Stratos (karlstratos@gmail.com)

#include "wsqloss.h"

void WSQLossOptimizer::Optimize(SMat W, SMat M, Eigen::MatrixXd *U,
				Eigen::MatrixXd *V) {
    // Will check that all dimensions match.
    string error_message = "Have W (" + to_string(W->rows) + "x" +
	to_string(W->cols) + "), M (" + to_string(M->rows) + "x" +
	to_string(M->cols) + "), U (" + to_string(U->rows()) + "x" +
	to_string(U->cols()) + "), and V (" + to_string(V->rows()) + "x" +
	to_string(V->cols());
    ASSERT(W->rows == M->rows && W->cols == M->cols, error_message);
    ASSERT(W->rows == U->cols() && W->cols == V->cols(), error_message);
    ASSERT(U->rows() == V->rows(), error_message);

    // Track how many "steps" have been taken in updating each parameter.
    vector<size_t> u_step(U->cols(), 1);
    vector<size_t> v_step(V->cols(), 1);

    // For readability, we will maintain the following convention:
    //    1. Index i denotes the current *row*.
    //    2. Index j denotes the current *column*.
    for (int epoch = 0; epoch < max_num_epochs_; ++epoch) {
	for (size_t j = 0; j < W->cols; ++j) {
	    size_t W_current_nonzero_index = W->pointr[j];
	    size_t W_next_start_nonzero_index = W->pointr[j + 1];
	    size_t M_current_nonzero_index = M->pointr[j];
	    size_t M_next_start_nonzero_index = M->pointr[j + 1];
	    ASSERT(W_current_nonzero_index == M_current_nonzero_index &&
		   W_next_start_nonzero_index == M_next_start_nonzero_index,
		   "Faulty indices");
	    size_t current_nonzero_index = W_current_nonzero_index;
	    size_t next_start_nonzero_index = W_next_start_nonzero_index;
	    while (current_nonzero_index < next_start_nonzero_index) {
		size_t W_row = W->rowind[current_nonzero_index];
		size_t M_row = M->rowind[current_nonzero_index];
		ASSERT(W_row == M_row, "Faulty indices");
		size_t i = W_row;
		double weight = W->value[current_nonzero_index];
		double value = M->value[current_nonzero_index];
		Eigen::VectorXd u = U->col(i);
		Eigen::VectorXd v = V->col(j);
		double predicted_value = u.dot(v);

		// Update the i-th column of U.
		Eigen::VectorXd u_negative_gradient =
		    weight * (value - predicted_value) * v
		    - regularization_term_ * u;
		U->col(i) += GetLearningRate(u_step[i]++) * u_negative_gradient;

		// Update the j-th column of V.
		Eigen::VectorXd v_negative_gradient =
		    weight * (value - predicted_value) * u
		    - regularization_term_ * v;
		V->col(j) += GetLearningRate(v_step[j]++) * v_negative_gradient;

		++current_nonzero_index;
	    }
	}
    }
}

double WSQLossOptimizer::GetLearningRate(size_t step) {
    double learning_rate = learning_rate_prior_ *
	pow(1.0 + learning_rate_prior_ * regularization_term_ * step, -0.75);
    return learning_rate;
}
