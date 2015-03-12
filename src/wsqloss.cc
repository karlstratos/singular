 // Author: Karl Stratos (stratos@cs.columbia.edu)

#include "wsqloss.h"

#include <iomanip>
#include <thread>

#include "util.h"

void solve_rows(const Eigen::MatrixXd &U, size_t num_threads, size_t thread_num,
		const WSQMap &mapping, Eigen::MatrixXd *V) {
    size_t block_size = ceil(double(V->rows()) / num_threads);  // Overestimate.
    size_t start_row = block_size * thread_num;
    size_t end_row = block_size * (thread_num + 1);
    if (end_row > V->rows()) { end_row = V->rows(); }
    LinearAlgebra lina;
    for (size_t row = start_row; row < end_row; ++row) {
	Eigen::MatrixXd U1 = Eigen::MatrixXd::Zero(U.rows(), U.cols());
	Eigen::RowVectorXd q = Eigen::RowVectorXd::Zero(U.cols());
	for (const auto &row_tuple: mapping.at(row)) {
	    size_t row = get<0>(row_tuple);
	    double value = get<1>(row_tuple);
	    double weight = get<2>(row_tuple);
	    U1.row(row) = weight * U.row(row);
	    q += value * U1.row(row);
	}
	Eigen::MatrixXd G = lina.ComputePinv(U.transpose() * U1);
	(*V).row(row) = q * G;
    }
}

void WSQLossOptimizer::Optimize(const WSQMap &col2row, const WSQMap &row2col,
				size_t max_num_epochs, size_t num_threads,
				Eigen::MatrixXd *U, Eigen::MatrixXd *V) {
    ASSERT(U->cols() == V->cols(), "U and V have different widths.");
    ASSERT(U->rows() == row2col.size(), "Row dimension mismatches.");
    ASSERT(V->rows() == col2row.size(), "Column dimension mismatches.");
    ASSERT(num_threads > 0, "Number of threads needs to be at least 1.");

    vector<thread> threads;
    double old_loss = ComputeLoss(col2row, *U, *V);
    cerr << fixed << setprecision(3);
    if (verbose_) { cerr << "Initial: " << old_loss << endl; }
    for (size_t epoch = 0; epoch < max_num_epochs; ++epoch) {
	// Solve rows of V in parallel.
	for (int thread_num = 0; thread_num < num_threads; ++thread_num) {
	    threads.push_back(thread(solve_rows, *U, num_threads, thread_num,
				     col2row, V));
	}
	for (auto &working_thread : threads) { working_thread.join(); }
	threads.clear();
	double loss = ComputeLoss(col2row, *U, *V);
	double reduction = old_loss - loss;
	old_loss = loss;
	if (verbose_) {
	    cerr << epoch + 1 << ". Solved V: " <<  loss << " ("
		 << reduction << ")" << endl;
	}
	if (reduction < kMinimumLossImprovement_) { break; }

	// Solve rows of U in parallel.
	for (int thread_num = 0; thread_num < num_threads; ++thread_num) {
	    threads.push_back(thread(solve_rows, *V, num_threads, thread_num,
				     row2col, U));
	}
	for (auto &working_thread : threads) { working_thread.join(); }
	threads.clear();
	loss = ComputeLoss(col2row, *U, *V);
	reduction = old_loss - loss;
	old_loss = loss;
	if (verbose_) {
	    cerr << epoch + 1 << ". Solved U: " <<  loss << " ("
		 << reduction << ")" << endl;
	}
	if (reduction < kMinimumLossImprovement_) { break; }
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
