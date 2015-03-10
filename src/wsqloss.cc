 // Author: Karl Stratos (stratos@cs.columbia.edu)

#include "wsqloss.h"

#include "util.h"

void WSQLossOptimizer::Optimize(const WSQMap &col2row, const WSQMap &row2col,
				Eigen::MatrixXd *U, Eigen::MatrixXd *V) {
    size_t rank = U->cols();
    ASSERT(V->cols() == rank, "U and V have different numbers of columns.");
    ASSERT(V->rows() == col2row.size(), "Column dimension mismatches.");
    ASSERT(U->rows() == row2col.size(), "Row dimension mismatches.");

    //(*U).setRandom();
    //(*V).setRandom();
    double loss = ComputeLoss(col2row, *U, *V);
    cout << "Initial loss: " << loss << endl;
    //for (size_t epoch = 0; epoch < max_num_epochs_; ++epoch) {
    for (size_t epoch = 0; epoch < 100; ++epoch) {
	// Fix U, optimize V.
	for (const auto &col_pair: col2row) {  // Solving for each row of V...
	    size_t col = col_pair.first;
	    Eigen::MatrixXd U1 = *U;
	    Eigen::RowVectorXd q = Eigen::RowVectorXd::Zero(rank);
	    for (const auto &row_tuple: col_pair.second) {
		size_t row = get<0>(row_tuple);
		double value = get<1>(row_tuple);
		double weight = get<2>(row_tuple);
		U1.row(row) *= weight;
		q += value * U1.row(row);
	    }
	    Eigen::MatrixXd G = ((*U).transpose() * U1).inverse();
	    (*V).row(col) = q * G;
	}
	loss = ComputeLoss(col2row, *U, *V);
	cout << "After optimizing V: " << loss << endl;

	// Fix V, optimize U.
	for (const auto &row_pair: row2col) {  // Solving for each row of U...
	    size_t row = row_pair.first;
	    Eigen::MatrixXd V1 = *V;
	    Eigen::RowVectorXd q = Eigen::RowVectorXd::Zero(rank);
	    for (const auto &col_tuple: row_pair.second) {
		size_t col = get<0>(col_tuple);
		double value = get<1>(col_tuple);
		double weight = get<2>(col_tuple);
		V1.row(col) *= weight;
		q += value * V1.row(col);
	    }
	    Eigen::MatrixXd G = ((*V).transpose() * V1).inverse();
	    (*U).row(row) = q * G;
	}
	loss = ComputeLoss(col2row, *U, *V);
	cout << "After optimizing U: " << loss << endl;
    }
}

double WSQLossOptimizer::ComputeLoss(const WSQMap &col2row,
				     const Eigen::MatrixXd &U,
				     const Eigen::MatrixXd &V) {
    double loss = 0.0;
    for (const auto &col_pair: col2row) {
	size_t col = col_pair.first;
	Eigen::VectorXd v = V.row(col);
	for (const auto &row_tuple: col_pair.second) {
	    size_t row = get<0>(row_tuple);
	    double value = get<1>(row_tuple);
	    double weight = get<2>(row_tuple);
	    loss += weight * pow(value - U.row(row).dot(v), 2);
	}
    }
    return loss;
}
