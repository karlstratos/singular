// Author: Karl Stratos (karlstratos@gmail.com)

#include "wsqloss.h"

void WSQLossOptimizer::Optimize(SMat W, SMat M, Eigen::MatrixXd *U,
				Eigen::MatrixXd *V) {
    // Check that all dimensions match.
    string error_message = "Have W (" + to_string(W->rows) + "x" +
	to_string(W->cols) + "), M (" + to_string(M->rows) + "x" +
	to_string(M->cols) + "), U (" + to_string(U->rows()) + "x" +
	to_string(U->cols()) + "), and V (" + to_string(V->rows()) + "x" +
	to_string(V->cols());
    ASSERT(W->rows == M->rows && W->cols == M->cols, error_message);
    ASSERT(U->rows() == V->rows(), error_message);
    ASSERT(W->rows == U->cols() && W->cols == V->cols(), error_message);

    // Track how many "steps" have been taken in updating each parameter.
    vector<size_t> u_step(U->cols(), 1);
    vector<size_t> v_step(V->cols(), 1);

    // For readability, we will maintain the following convention:
    //    1. Indices i denote rows of W/M    => columns of U.
    //    2. Indices j denote columns of W/M => columns of V.
    for (int epoch = 0; epoch < max_num_epochs_; ++epoch) {
	cout << "Starting epoch " << epoch + 1 << endl;
	for (size_t j = 0; j < W->cols; ++j) {
	    ASSERT(W->pointr[j] == M->pointr[j] &&
		   W->pointr[j + 1] == M->pointr[j + 1], "Faulty indices");
	    size_t current_nonzero_index = W->pointr[j];
	    size_t next_start_nonzero_index = W->pointr[j + 1];
	    Eigen::VectorXd v = V->col(j);
	    while (current_nonzero_index < next_start_nonzero_index) {
		ASSERT(W->rowind[current_nonzero_index] ==
		       M->rowind[current_nonzero_index], "Faulty indices");
		size_t i = W->rowind[current_nonzero_index];
		double weight = W->value[current_nonzero_index];
		double value = M->value[current_nonzero_index];
		Eigen::VectorXd u = U->col(i);
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
	double wsq_loss = ComputeWSQLoss(W, M, *U, *V);
	cerr << "Epoch " << epoch + 1 << ": " << wsq_loss << endl;
    }
}

double WSQLossOptimizer::ComputeWSQLoss(SMat W, SMat M,
					const Eigen::MatrixXd &U,
					const Eigen::MatrixXd &V) {
    double wsq_loss = 0.0;
    //   =  sum_{i=1...d, j=1...d'} W_{i,j} * (M_{i,j} - U_i' V_j)^2 +
    //               sum_{i=1...d}  regularization_term_ * ||U_i||^2 +
    //               sum_{j=1...d'} regularization_term_ * ||V_j||^2 +

    unordered_map<size_t, bool> added_indices_i;
    for (size_t j = 0; j < W->cols; ++j) {
	ASSERT(W->pointr[j] == M->pointr[j] &&
	       W->pointr[j + 1] == M->pointr[j + 1], "Faulty indices");
	size_t current_nonzero_index = W->pointr[j];
	size_t next_start_nonzero_index = W->pointr[j + 1];
	Eigen::VectorXd v = V.col(j);

	// Adding: regularization_term_ * ||V_j||^2
	wsq_loss += regularization_term_ * v.squaredNorm();

	while (current_nonzero_index < next_start_nonzero_index) {
	    ASSERT(W->rowind[current_nonzero_index] ==
		   M->rowind[current_nonzero_index], "Faulty indices");
	    size_t i = W->rowind[current_nonzero_index];
	    double weight = W->value[current_nonzero_index];
	    double value = M->value[current_nonzero_index];
	    Eigen::VectorXd u = U.col(i);
	    double predicted_value = u.dot(v);

	    if (added_indices_i.find(i) == added_indices_i.end()) {
		// Adding (without repetition): regularization_term_ * ||U_i||^2
		wsq_loss += regularization_term_ * u.squaredNorm();
		added_indices_i[i] = true;
	    }

	    // Adding: W_{i,j} * (M_{i,j} - U_i' V_j)^2
	    wsq_loss += weight * pow(value - predicted_value, 2);

	    ++current_nonzero_index;
	}
    }
    return wsq_loss;
}

double WSQLossOptimizer::GetLearningRate(size_t step) {
    double learning_rate = learning_rate_prior_ *
	pow(1.0 + learning_rate_prior_ * regularization_term_ * step, -0.75);
    return learning_rate;
}
