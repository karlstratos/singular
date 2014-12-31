// Author: Karl Stratos (karlstratos@gmail.com)
//
// A SparseCCASolver efficiently performs canonical correlation analysis (CCA)
// on data with sparse features. A central trick is to skip centering (and
// thereby preserve sparsity) by assuming that the mean of each feature is
// zero. This assumption is justified if (i) the number of samples is large
// and (ii) each feature type occurs infrequently and takes binary (or some
// other small-valued) value. Another central trick is to assume that features
// are independent so that the square-root inverse covariance matrix is
// diagonal. CCA needs to perform an SVD on a scaled cross-coveriance matrix:
//
//           Omega = Cov(X,X)^{-1/2} Cov(X,Y) Cov(Y,Y)^{-1/2}
//
// (Typically, some smoothing is done for the matrix M for square-root inverse
// to make it better conditioned: e.g., M + aI for some small constant a.)
// Here, the scaling matrices will be dense (even with the zero mean assumption)
// and thus computationally challenging. If we make the independence assumption,
// however, this becomes simple entrywise scaling of Cov(X,Y).
// The high-level algorithm is as follows:
//   1. Compute the sparse matrix Omega above with appropriate assumptions.
//   2. Do an SVD on Omega and compute the m left and right singular vectors
//      corresponding to the largest singular values.
//   3. Return the singular vectors scaled by square-root inverse covariance
//      matrices. These are the linear transformations for the two views.

#ifndef SPARSECCA_H
#define SPARSECCA_H

#include <Eigen/Dense>

#include "sparsesvd.h"

class SparseCCASolver {
public:
    // Initializes the solver with the given dimension and smoothing term.
    SparseCCASolver(size_t cca_dim, double smoothing_term) :
	cca_dim_(cca_dim), smoothing_term_(smoothing_term) { }

    // Computes CCA transformations from given cross-covariance and (diagonal
    // approximation) covariance matrices of (near) zero-mean variables X and Y.
    // Because of cancellation, you can safely skip normalizing the covariance
    // values. That is, you can simple use (with l=1...n examples):
    //   covariance_xy[j][i] = sum_{l=1}^n x^(l)_i * y^(l)_j
    //   variance_x[i]       = sum_{l=1}^n x^(l)_i^2
    //   variance_y[i]       = sum_{l=1}^n y^(l)_i^2
    // where you skipped dividing by (n-1) in each term.
    void PerformCCA(unordered_map<size_t, unordered_map<size_t, double> >
		    *covariance_xy,
		    const unordered_map<size_t, double> &variance_x,
		    const unordered_map<size_t, double> &variance_y);

    // Computes CCA transformations from given files containing covariance
    // information. This may be more appropriate for large-scale CCA since it
    // avoids loading two copies of the matrix for SVD.
    void PerformCCA(const string &covariance_xy_file,
		    const string &variance_x_file,
		    const string &variance_y_file);

    // Computes CCA transformations for two views from given examples. Each pair
    // of examples in the two sequences represents two "views" of an object.
    void PerformCCA(const vector<unordered_map<size_t, double> > &examples_x,
		    const vector<unordered_map<size_t, double> > &examples_y);

    // Sets the dimension of the CCA subspace.
    void set_cca_dim(size_t cca_dim) { cca_dim_ = cca_dim; }

    // Sets the smoothing term for calculating the correlation matrix.
    void set_smoothing_term(size_t smoothing_term) {
	smoothing_term_ = smoothing_term;
    }

    // Returns the dimension of the CCA subspace.
    size_t cca_dim() { return cca_dim_; }

    // Returns the actually computed rank. This is equal to the CCA dimension
    // unless the correlation matrix has a smaller rank.
    size_t rank() { return rank_; }

    // Returns the smoothing term for calculating the correlation matrix.
    size_t smoothing_term() { return smoothing_term_; }

    // Returns the CCA tranformation for the first view X.
    Eigen::MatrixXd *cca_transformation_x() { return &cca_transformation_x_; }

    // Returns the CCA tranformation for the second view Y.
    Eigen::MatrixXd *cca_transformation_y() { return &cca_transformation_y_; }

    // Returns the CCA correlation values.
    Eigen::VectorXd *cca_correlations() { return &cca_correlations_; }

private:
    // Extracts scaled singular vectors from an SVD solver.
    void ExtractScaledSingularVectors(
	const SparseSVDSolver &svd_solver,
	const unordered_map<size_t, double> &variance_x,
	const unordered_map<size_t, double> &variance_y);

    // Dimension of the CCA subspace.
    size_t cca_dim_;

    // The actually computed rank. This is equal to the CCA dimension unless
    // the correlation matrix has a smaller rank.
    size_t rank_ = 0;

    // Smoothing term for calculating the correlation matrix.
    double smoothing_term_;

    // CCA tranformation for the first view X: (dimension) x |X|.
    Eigen::MatrixXd cca_transformation_x_;

    // CCA tranformation for the second view Y: (dimension) x |Y|.
    Eigen::MatrixXd cca_transformation_y_;

    // CCA correlations (singular values): (dimension) x 1. These are not
    // necessarily proper correlation values due to approximations.
    Eigen::VectorXd cca_correlations_;
};

#endif  // SPARSECCA_H
