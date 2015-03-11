 // Author: Karl Stratos (stratos@cs.columbia.edu)

#include "wsqloss.h"

#include "util.h"

void WSQLossOptimizer::Optimize(const WSQMap &col2row, const WSQMap &row2col,
				Eigen::MatrixXd *U, Eigen::MatrixXd *V) {
    size_t rank = U->cols();
    ASSERT(V->cols() == rank, "U and V have different numbers of columns.");
    ASSERT(V->rows() == col2row.size(), "Column dimension mismatches.");
    ASSERT(U->rows() == row2col.size(), "Row dimension mismatches.");


    unordered_map<size_t, unordered_map<size_t, pair<double, double> > > table;
    for (const auto &col_pair: col2row) {
	size_t col = col_pair.first;
	for (const auto &row_tuple: col_pair.second) {
	    size_t row = get<0>(row_tuple);
	    double value = get<1>(row_tuple);
	    double weight = get<2>(row_tuple);
	    table[row][col] = make_pair(value, weight);
	}
    }

    for (const auto &row_pair: row2col) {
	size_t row = row_pair.first;
	for (const auto &col_tuple: row_pair.second) {
	    size_t col = get<0>(col_tuple);
	    double value = get<1>(col_tuple);
	    double weight = get<2>(col_tuple);

	    double value2 = table[row][col].first;
	    double weight2 = table[row][col].second;
	    assert(fabs(value - value2) < 1e-13);
	    assert(fabs(weight - weight2) < 1e-13);
	}
    }
    //exit(0);

    //(*U).setRandom();
    //(*V).setRandom();
    double loss = ComputeLoss(col2row, row2col, *U, *V);
    cout << "Initial loss: " << loss << endl;
    //exit(0);
    //for (size_t epoch = 0; epoch < max_num_epochs_; ++epoch) {
    for (size_t epoch = 0; epoch < 10; ++epoch) {
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
	    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
		(*U).transpose() * U1 +
		0.1 * Eigen::MatrixXd::Identity(rank, rank));
	    /*
	    cout << "Condition: " << svd.singularValues()(0) /
		svd.singularValues()(rank-1) << endl;
	    */
	    Eigen::MatrixXd G = (
		(*U).transpose() * U1 +
		0.1 * Eigen::MatrixXd::Identity(rank, rank)).inverse();
	    (*V).row(col) = q * G;
	}
	loss = ComputeLoss(col2row, row2col, *U, *V);
	cout << "After optimizing V: " << loss << endl;
	//exit(0);

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
	    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
		(*V).transpose() * V1 +
		0.1 * Eigen::MatrixXd::Identity(rank, rank));
	    /*
	    cout << "Condition: " << svd.singularValues()(0) /
		svd.singularValues()(rank-1) << endl;
	    */
	    Eigen::MatrixXd G = (
		(*V).transpose() * V1 +
		0.1 * Eigen::MatrixXd::Identity(rank, rank)).inverse();
	    (*U).row(row) = q * G;
	}
	loss = ComputeLoss(col2row, row2col, *U, *V);
	cout << "After optimizing U: " << loss << endl;
    }
}

double WSQLossOptimizer::ComputeLoss(const WSQMap &col2row,
				     const WSQMap &row2col,
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
	    loss += weight * pow(value - U.row(row).dot(V.row(col)), 2);
	}
    }

    for (const auto &col_pair: col2row) {
	size_t col = col_pair.first;
	loss += 0.1 * V.row(col).norm();
    }
    for (const auto &row_pair: row2col) {
	size_t row = row_pair.first;
	loss += 0.1 * U.row(row).norm();
    }


    double loss2 = 0.0;
    for (const auto &row_pair: row2col) {
	size_t row = row_pair.first;
	Eigen::VectorXd u = U.row(row);
	for (const auto &col_tuple: row_pair.second) {
	    size_t col = get<0>(col_tuple);
	    double value = get<1>(col_tuple);
	    double weight = get<2>(col_tuple);
	    loss2 += weight * pow(value - u.dot(V.row(col)), 2);
	}
    }

    for (const auto &col_pair: col2row) {
	size_t col = col_pair.first;
	loss2 += 0.1 * V.row(col).norm();
    }
    for (const auto &row_pair: row2col) {
	size_t row = row_pair.first;
	loss2 += 0.1 * U.row(row).norm();
    }

    //cout << loss << " " << loss2 << endl;
    assert(fabs(loss - loss2) < 1e-10);

    return loss;
}
