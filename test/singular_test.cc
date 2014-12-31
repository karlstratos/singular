// Author: Karl Stratos (karlstratos@gmail.com)
//
// Check the correctness of the code in the source directory.

#include <random>

#include "gtest/gtest.h"
#include "../src/sparsecca.h"
#include "../src/sparsesvd.h"
#include "../src/wordrep.h"

// Test class that provides a dense random matrix.
class DenseRandomMatrix : public testing::Test {
protected:
    virtual void SetUp() {
	for (size_t column_index = 0; column_index < num_columns_;
	     ++column_index) {
	    for (size_t row_index = 0; row_index < num_rows_; ++row_index) {
		random_device device;
		default_random_engine engine(device());
		normal_distribution<double> normal(0.0, 1.0);
		double random_value = normal(engine);
		column_map_[column_index][row_index] = random_value;
	    }
	}
    }

    virtual void TearDown() { }

    size_t num_rows_ = 5;
    size_t num_columns_ = 4;
    size_t full_rank_ = min(num_rows_, num_columns_);
    unordered_map<size_t, unordered_map<size_t, double> > column_map_;
    SparseSVDSolver sparsesvd_solver_;
};

// Tests a full SVD a random (full-rank) matrix.
TEST_F(DenseRandomMatrix, DecomposeFully) {
    sparsesvd_solver_.LoadSparseMatrix(column_map_);
    sparsesvd_solver_.SolveSparseSVD(full_rank_);  // Full SVD.
    EXPECT_EQ(full_rank_, sparsesvd_solver_.rank());
}

// Test class that provides an identity matrix.
class IdentityMatrix : public testing::Test {
protected:
    virtual void SetUp() {
	for (size_t column_index = 0; column_index < num_columns_;
	     ++column_index) {
	    vector<pair<size_t, double> > row_index_value_pairs;
	    for (size_t row_index = 0; row_index < num_rows_; ++row_index) {
		if (row_index == column_index) {
		    column_map_[column_index][row_index] = 1.0;
		}
	    }
	}
    }

    virtual void TearDown() { }

    size_t num_rows_ = 4;
    size_t num_columns_ = num_rows_;
    size_t full_rank_ = num_rows_;
    unordered_map<size_t, unordered_map<size_t, double> > column_map_;
    SparseSVDSolver sparsesvd_solver_;
};

// Confirms that SVDLIBC breaks without eigengaps (e.g., an identity matrix).
TEST_F(IdentityMatrix, BreaksWithoutEigengaps) {
    sparsesvd_solver_.LoadSparseMatrix(column_map_);
    sparsesvd_solver_.SolveSparseSVD(full_rank_);
    EXPECT_NE(full_rank_, sparsesvd_solver_.rank());
}

// Confirms that SVDLIBC breaks even with a nonzero eigengap if small.
TEST_F(IdentityMatrix, BreaksEvenWithANonzeroEigengap) {
    // Introduce a nonzero eigengap in an identity matrix.
    column_map_[0][0] = 1.0000001;

    sparsesvd_solver_.LoadSparseMatrix(column_map_);
    sparsesvd_solver_.SolveSparseSVD(full_rank_);
    EXPECT_NE(full_rank_, sparsesvd_solver_.rank());
}

// Confirms that SVDLIBC works correctly with some eigengaps.
TEST_F(IdentityMatrix, DoesNotBreakWithEigengaps) {
    // Introduce eigengaps in an identity matrix.
    size_t value = num_rows_;
    for (size_t i = 0; i < num_rows_; ++i) {
	column_map_[i][i] = value--;
    }

    sparsesvd_solver_.LoadSparseMatrix(column_map_);
    sparsesvd_solver_.SolveSparseSVD(full_rank_);
    EXPECT_EQ(full_rank_, sparsesvd_solver_.rank());
}

// Test class that provides a sparse matrix with empty columns.
class SparseMatrixWithEmptyColumns : public testing::Test {
protected:
    virtual void SetUp() {
	//      Empty columns
	//        |     |
	//        |     |
	//        v     v
	//
	//     0  0  1  0
	//     0  0  0  0
	//     2  0  3  0
	//     0  0  4  0
	column_map_[0][2] = 2.0;
	column_map_[2][0] = 1.0;
	column_map_[2][2] = 3.0;
	column_map_[2][3] = 4.0;
    }

    virtual void TearDown() { }

    size_t num_rows_ = 4;
    size_t num_columns_ = 4;
    unordered_map<size_t, unordered_map<size_t, double> > column_map_;
    SparseSVDSolver sparsesvd_solver_;
    double tol_ = 1e-4;
};

// Confirms that SVDLIBC works correctly on the matrix.
TEST_F(SparseMatrixWithEmptyColumns, CorrectnessOfSVDLIBC) {
    sparsesvd_solver_.LoadSparseMatrix(column_map_);
    sparsesvd_solver_.SolveSparseSVD(2);
    EXPECT_EQ(2, sparsesvd_solver_.rank());
    EXPECT_NEAR(5.2469, fabs(*(sparsesvd_solver_.singular_values() + 0)), tol_);
    EXPECT_NEAR(1.5716, fabs(*(sparsesvd_solver_.singular_values() + 1)), tol_);
}

// Confirms that writing and loading this sparse matrix is correct.
TEST_F(SparseMatrixWithEmptyColumns, WriteAndLoad) {
    // Write the matrix to a temporary file.
    string temp_file_path = tmpnam(nullptr);
    sparsesvd_solver_.WriteSparseMatrix(column_map_, temp_file_path);

    // Load the matrix from that file.
    sparsesvd_solver_.LoadSparseMatrix(temp_file_path);

    // Solve SVD and check the result.
    sparsesvd_solver_.SolveSparseSVD(2);
    EXPECT_EQ(2, sparsesvd_solver_.rank());
    EXPECT_NEAR(5.2469, fabs(*(sparsesvd_solver_.singular_values() + 0)), tol_);
    EXPECT_NEAR(1.5716, fabs(*(sparsesvd_solver_.singular_values() + 1)), tol_);
}

// Test class that provides a simple example for CCA.
class CCASimpleExample : public testing::Test {
protected:
    virtual void SetUp() {
	covariance_xy_[0][0] = 3.0;
	covariance_xy_[1][1] = 1.0;
	covariance_xy_[2][3] = 1.0;
	covariance_xy_[2][2] = 1.0;
	covariance_xy_[3][1] = 1.0;
	covariance_xy_[4][1] = 1.0;
	covariance_xy_[5][4] = 1.0;

	variance_x_[0] = 3.0;
	variance_x_[1] = 3.0;
	variance_x_[2] = 1.0;
	variance_x_[3] = 1.0;
	variance_x_[4] = 1.0;

	variance_y_[0] = 3.0;
	variance_y_[1] = 1.0;
	variance_y_[2] = 2.0;
	variance_y_[3] = 1.0;
	variance_y_[4] = 1.0;
	variance_y_[5] = 1.0;
    }

    virtual void TearDown() { }

    unordered_map<size_t, unordered_map<size_t, double> > covariance_xy_;
    unordered_map<size_t, double> variance_x_;
    unordered_map<size_t, double> variance_y_;
    double tol_ = 1e-3;
};

// Checks the CCA result with dimension 2 and smoothing 1.
TEST_F(CCASimpleExample, Dimension2Smoothing1) {
    SparseCCASolver sparsecca_solver(2, 1.0);
    sparsecca_solver.PerformCCA(&covariance_xy_, variance_x_, variance_y_);

    Eigen::VectorXd correlation_values = *sparsecca_solver.cca_correlations();
    EXPECT_NEAR(0.7500, correlation_values(0), tol_);
    EXPECT_NEAR(0.6125, correlation_values(1), tol_);
}

// Checks the CCA result with dimension 2 and smoothing 1 using samples. These
// samples result in the same covariance and variance values as above.
TEST_F(CCASimpleExample, SamplesDimension2Smoothing1) {
    SparseCCASolver sparsecca_solver(2, 1.0);
    vector<unordered_map<size_t, double> > examples_x;
    vector<unordered_map<size_t, double> > examples_y;
    unordered_map<size_t, double> x;
    unordered_map<size_t, double> y;
    x[0] = 1.0;
    y[0] = 1.0;
    examples_x.push_back(x);
    examples_y.push_back(y);

    x.clear();
    y.clear();
    x[1] = 1.0;
    y[1] = 1.0;
    examples_x.push_back(x);
    examples_y.push_back(y);

    x.clear();
    y.clear();
    x[2] = 1.0;
    y[2] = 1.0;
    examples_x.push_back(x);
    examples_y.push_back(y);

    x.clear();
    y.clear();
    x[0] = 1.0;
    y[0] = 1.0;
    examples_x.push_back(x);
    examples_y.push_back(y);

    x.clear();
    y.clear();
    x[1] = 1.0;
    y[3] = 1.0;
    examples_x.push_back(x);
    examples_y.push_back(y);

    x.clear();
    y.clear();
    x[3] = 1.0;
    y[2] = 1.0;
    examples_x.push_back(x);
    examples_y.push_back(y);

    x.clear();
    y.clear();
    x[0] = 1.0;
    y[0] = 1.0;
    examples_x.push_back(x);
    examples_y.push_back(y);

    x.clear();
    y.clear();
    x[1] = 1.0;
    y[4] = 1.0;
    examples_x.push_back(x);
    examples_y.push_back(y);

    x.clear();
    y.clear();
    x[4] = 1.0;
    y[5] = 1.0;
    examples_x.push_back(x);
    examples_y.push_back(y);

    sparsecca_solver.PerformCCA(examples_x, examples_y);

    Eigen::VectorXd correlation_values = *sparsecca_solver.cca_correlations();
    EXPECT_NEAR(0.7500, correlation_values(0), tol_);
    EXPECT_NEAR(0.6125, correlation_values(1), tol_);
}

// Test class that provides a simple corpus for inducing CCA word vectors.
class CanonWordSimpleExample : public testing::Test {
protected:
    virtual void SetUp() {
	temp_file_path_ = tmpnam(nullptr);
	ofstream temp_file(temp_file_path_, ios::out);
	temp_file << "a b c" << endl;
	temp_file << "a b d" << endl;
	temp_file << "a b e" << endl;

	temp_output_directory_ = tmpnam(nullptr);
    }

    virtual void TearDown() { }

    string temp_file_path_;
    string temp_output_directory_;
    StringManipulator string_manipulator_;
    string line_;
    vector<string> tokens_;
    double tol_ = 1e-4;
};

// Only checks counts with cutoff 0 and window size 2.
TEST_F(CanonWordSimpleExample, OnlyCheckCountsCutoff0WindowSize2) {
    CanonWord canonword(temp_output_directory_);
    canonword.set_rare_cutoff(0);
    canonword.set_window_size(2);
    canonword.ExtractStatistics(temp_file_path_);

    // Check against the true counts.
    unordered_map<string, unordered_map<string, size_t> >
	true_count_word_context;
    unordered_map<string, size_t> true_count_word;
    unordered_map<string, size_t> true_count_context;

    // a b c a b d a b e
    true_count_word_context["w(1)=b"]["a"] = 3;
    true_count_word_context["w(1)=c"]["b"] = 1;
    true_count_word_context["w(1)=a"]["c"] = 1;
    true_count_word_context["w(1)=d"]["b"] = 1;
    true_count_word_context["w(1)=a"]["d"] = 1;
    true_count_word_context["w(1)=e"]["b"] = 1;
    true_count_word_context["w(1)=" + canonword.kBufferString()]["e"] = 1;
    true_count_word["a"] = 3;
    true_count_word["b"] = 3;
    true_count_word["c"] = 1;
    true_count_word["d"] = 1;
    true_count_word["e"] = 1;
    true_count_context["w(1)=b"] = 3;
    true_count_context["w(1)=c"] = 1;
    true_count_context["w(1)=a"] = 2;
    true_count_context["w(1)=d"] = 1;
    true_count_context["w(1)=e"] = 1;
    true_count_context["w(1)=" + canonword.kBufferString()] = 1;

    ifstream word_context_file(canonword.CountWordContextPath(), ios::in);
    int col = -1;
    while (word_context_file.good()) {
	getline(word_context_file, line_);
	string_manipulator_.split(line_, " ", &tokens_);
	if (tokens_.size() == 1) { ++col; }
	if (tokens_.size() == 2) {
	    string word_string = canonword.word_num2str(stoi(tokens_[0]));
	    string context_string = canonword.context_num2str(col);
	    size_t count = stoi(tokens_[1]);
	    EXPECT_EQ(true_count_word_context[context_string][word_string],
		      count);
	}
    }

    ifstream word_file(canonword.CountWordPath(), ios::in);
    Word word = 0;
    while (word_file.good()) {
	getline(word_file, line_);
	string_manipulator_.split(line_, " ", &tokens_);
	if (tokens_.size() == 0) { continue; }
	string word_string = canonword.word_num2str(word++);
	size_t count = stoi(tokens_[0]);
	EXPECT_EQ(true_count_word[word_string], count);
    }

    ifstream context_file(canonword.CountContextPath(), ios::in);
    Context context = 0;
    while (context_file.good()) {
	getline(context_file, line_);
	string_manipulator_.split(line_, " ", &tokens_);
	if (tokens_.size() == 0) { continue; }
	string context_string = canonword.context_num2str(context++);
	size_t count = stoi(tokens_[0]);
	EXPECT_EQ(true_count_context[context_string], count);
    }
}

// Checks that SVDLIBC completely fails when the gap between the largest
// singular values is small, which is the case with this example when no
// smoothing is applied.
TEST_F(CanonWordSimpleExample, SVDLIBCFailsWithoutSmoothingCutoff0WindowSize2) {
    CanonWord canonword(temp_output_directory_);
    canonword.set_rare_cutoff(0);
    canonword.set_window_size(2);
    canonword.set_cca_dim(2);
    canonword.set_smoothing_term(0.0);
    canonword.ExtractStatistics(temp_file_path_);
    canonword.InduceLexicalRepresentations();

    // The correlation matrix is:
    //    1.0000 0.0000 0.0000 0.0000 0.0000 0.0000
    //    0.0000 0.5774 0.0000 0.5774 0.5774 0.0000
    //    0.0000 0.0000 0.7071 0.0000 0.0000 0.0000
    //    0.0000 0.0000 0.7071 0.0000 0.0000 0.0000
    //    0.0000 0.0000 0.0000 0.0000 0.0000 1.0000
    // Its two largest singular values are too close (1 and 1), so SVDLIBC
    // gives rubbish singular vectors. Matlab's top 2 left singular vectors are:
    //    -0.0000    0.2594
    //    -1.0000   -0.0000
    //    -0.0000    0.0000
    //    -0.0000    0.0000
    //    -0.0000   -0.9658
    // SVDLIBC's top 2 left singular vectors are:
    //    -0.0000  0.0001
    //     0.8394  0.2564
    //     0.3814 -0.2012
    //     0.3814 -0.2012
    //    -0.0668  0.9237
    // The model does PCA on this wrong matrix after row normalization.
    Eigen::VectorXd singular_values = *canonword.singular_values();
    // These singular values are actually better than usual: SVDLIBC can return
    // rank 0 and zero singular values when it has this problem!
    EXPECT_NEAR(1.0000, singular_values(0), tol_);
    EXPECT_NEAR(1.0000, singular_values(1), tol_);

    unordered_map<string, Eigen::VectorXd> vectors =
	*canonword.wordvectors();
    Eigen::VectorXd vector1 = vectors[canonword.word_num2str(1)];
    EXPECT_NEAR(0.3338, fabs(vector1(0)), tol_);  // Wrong.
    EXPECT_NEAR(0.4164, fabs(vector1(1)), tol_);  // Wrong.
}

// Checks that SVDLIBC doesn't fail if some smoothing is applied to make the gap
// between the largest singular values larger.
TEST_F(CanonWordSimpleExample, SVDLIBCSucceedsWithSmoothingCutoff0WindowSize2) {
    CanonWord canonword(temp_output_directory_);
    canonword.set_rare_cutoff(0);
    canonword.set_window_size(2);
    canonword.set_cca_dim(2);
    canonword.set_smoothing_term(1.0);  // Add 1 to dividing counts.
    canonword.ExtractStatistics(temp_file_path_);
    canonword.InduceLexicalRepresentations();

    // The correlation matrix is now:
    //    0.7500 0.0000 0.0000 0.0000 0.0000 0.0000
    //    0.0000 0.3536 0.0000 0.3536 0.3536 0.0000
    //    0.0000 0.0000 0.4082 0.0000 0.0000 0.0000
    //    0.0000 0.0000 0.4082 0.0000 0.0000 0.0000
    //    0.0000 0.0000 0.0000 0.0000 0.0000 0.5000
    // Its two largest singular values are not as close (0.7500 and 0.6124),
    // so SVDLIBC gives correct top 2 left singular vectors, which are:
    //    1.0000 0.0000
    //    0.0000 1.0000
    //    0.0000 0.0000
    //    0.0000 0.0000
    //    0.0000 0.0000
    // HOWEVER, the "zero" values are not truly zero but some tiny values. After
    // row normalization, they blow up to arbitrary values as follows:
    //    1.0000    0.0000
    //   -0.0000    1.0000
    //    0.7742    0.6329
    //    0.7742    0.6329
    //   -0.9976    0.0690
    // This affects the post-processing PCA step. We will take this as ground
    // truth (Matlab gives a different blow-up) and do PCA on this matrix, which
    // gives:
    //   0.6437   -0.5287
    //  -0.2596    0.5594
    //   0.4774    0.1223
    //   0.4774    0.1223
    //  -1.3390   -0.2754
    Eigen::VectorXd singular_values = *canonword.singular_values();
    EXPECT_NEAR(0.7500, singular_values(0), tol_);
    EXPECT_NEAR(0.6124, singular_values(1), tol_);

    unordered_map<string, Eigen::VectorXd> vectors =
	*canonword.wordvectors();
    Eigen::VectorXd vector1 = vectors[canonword.word_num2str(1)];
    EXPECT_NEAR(0.2596, fabs(vector1(0)), tol_);
    EXPECT_NEAR(0.5594, fabs(vector1(1)), tol_);
}

// Only checks counts with cutoff 1 and window size 3.
TEST_F(CanonWordSimpleExample, OnlyCheckCountsCutoff1WindowSize3) {
    CanonWord canonword(temp_output_directory_);
    canonword.set_rare_cutoff(1);
    canonword.set_window_size(3);
    canonword.set_cca_dim(2);
    canonword.set_smoothing_term(0.0);
    canonword.ExtractStatistics(temp_file_path_);

    // Check against the true counts.
    unordered_map<string, unordered_map<string, size_t> >
	true_count_word_context;
    unordered_map<string, size_t> true_count_word;
    unordered_map<string, size_t> true_count_context;

    // a b <?> a b <?> a b <?>
    true_count_word_context["w(-1)=" + canonword.kBufferString()]["a"] = 1;
    true_count_word_context["w(1)=b"]["a"] = 3;
    true_count_word_context["w(-1)=a"]["b"] = 3;
    true_count_word_context["w(1)=" + canonword.kRareString()]["b"] = 3;
    true_count_word_context["w(-1)=b"][canonword.kRareString()] = 3;
    true_count_word_context["w(1)=a"][canonword.kRareString()] = 2;
    true_count_word_context["w(-1)=" + canonword.kRareString()]["a"] = 2;
    true_count_word_context["w(1)=" + canonword.kBufferString()][
	canonword.kRareString()] = 1;
    true_count_word["a"] = 3;
    true_count_word["b"] = 3;
    true_count_word[canonword.kRareString()] = 3;
    true_count_context["w(-1)=" + canonword.kBufferString()] = 1;
    true_count_context["w(1)=b"] = 3;
    true_count_context["w(-1)=a"] = 3;
    true_count_context["w(1)=" + canonword.kRareString()] = 3;
    true_count_context["w(-1)=b"] = 3;
    true_count_context["w(1)=a"] = 2;
    true_count_context["w(-1)=" + canonword.kRareString()] = 2;
    true_count_context["w(1)=" + canonword.kBufferString()] = 1;

    ifstream word_context_file(canonword.CountWordContextPath(), ios::in);
    int col = -1;
    while (word_context_file.good()) {
	getline(word_context_file, line_);
	string_manipulator_.split(line_, " ", &tokens_);
	if (tokens_.size() == 1) { ++col; }
	if (tokens_.size() == 2) {
	    string word_string = canonword.word_num2str(stoi(tokens_[0]));
	    string context_string = canonword.context_num2str(col);
	    size_t count = stoi(tokens_[1]);
	    EXPECT_EQ(true_count_word_context[context_string][word_string],
		      count);
	}
    }

    ifstream word_file(canonword.CountWordPath(), ios::in);
    Word word = 0;
    while (word_file.good()) {
	getline(word_file, line_);
	string_manipulator_.split(line_, " ", &tokens_);
	if (tokens_.size() == 0) { continue; }
	string word_string = canonword.word_num2str(word++);
	size_t count = stoi(tokens_[0]);
	EXPECT_EQ(true_count_word[word_string], count);
    }

    ifstream context_file(canonword.CountContextPath(), ios::in);
    Context context = 0;
    while (context_file.good()) {
	getline(context_file, line_);
	string_manipulator_.split(line_, " ", &tokens_);
	if (tokens_.size() == 0) { continue; }
	string context_string = canonword.context_num2str(context++);
	size_t count = stoi(tokens_[0]);
	EXPECT_EQ(true_count_context[context_string], count);
    }
}

// Only checks counts with cutoff 1 and window size 3: use the sentence-per-line
// mode.
TEST_F(CanonWordSimpleExample,
       OnlyCheckCountsCutoff1WindowSize3SentencePerLine) {
    CanonWord canonword(temp_output_directory_);
    canonword.set_rare_cutoff(1);
    canonword.set_window_size(3);
    canonword.set_sentence_per_line(true);
    canonword.ExtractStatistics(temp_file_path_);

    // Check against the true counts.
    unordered_map<string, unordered_map<string, size_t> >
	true_count_word_context;
    unordered_map<string, size_t> true_count_word;
    unordered_map<string, size_t> true_count_context;

    // <!> a b <?> <!>
    // <!> a b <?> <!>
    // <!> a b <?> <!>
    true_count_word_context["w(-1)=" + canonword.kBufferString()]["a"] = 3;
    true_count_word_context["w(1)=b"]["a"] = 3;
    true_count_word_context["w(-1)=a"]["b"] = 3;
    true_count_word_context["w(1)=" + canonword.kRareString()]["b"] = 3;
    true_count_word_context["w(-1)=b"][canonword.kRareString()] = 3;
    true_count_word_context["w(1)=" + canonword.kBufferString()][
	canonword.kRareString()] = 3;
    true_count_word["a"] = 3;
    true_count_word["b"] = 3;
    true_count_word[canonword.kRareString()] = 3;
    true_count_context["w(-1)=" + canonword.kBufferString()] = 3;
    true_count_context["w(1)=b"] = 3;
    true_count_context["w(-1)=a"] = 3;
    true_count_context["w(1)=" + canonword.kRareString()] = 3;
    true_count_context["w(-1)=b"] = 3;
    true_count_context["w(1)=" + canonword.kBufferString()] = 3;

    ifstream word_context_file(canonword.CountWordContextPath(), ios::in);
    int col = -1;
    while (word_context_file.good()) {
	getline(word_context_file, line_);
	string_manipulator_.split(line_, " ", &tokens_);
	if (tokens_.size() == 1) { ++col; }
	if (tokens_.size() == 2) {
	    string word_string = canonword.word_num2str(stoi(tokens_[0]));
	    string context_string = canonword.context_num2str(col);
	    size_t count = stoi(tokens_[1]);
	    EXPECT_EQ(true_count_word_context[context_string][word_string],
		      count);
	}
    }

    ifstream word_file(canonword.CountWordPath(), ios::in);
    Word word = 0;
    while (word_file.good()) {
	getline(word_file, line_);
	string_manipulator_.split(line_, " ", &tokens_);
	if (tokens_.size() == 0) { continue; }
	string word_string = canonword.word_num2str(word);
	size_t count = stoi(tokens_[0]);
	EXPECT_EQ(true_count_word[word_string], count);
    }

    ifstream context_file(canonword.CountContextPath(), ios::in);
    Context context = 0;
    while (context_file.good()) {
	getline(context_file, line_);
	string_manipulator_.split(line_, " ", &tokens_);
	if (tokens_.size() == 0) { continue; }
	string context_string = canonword.context_num2str(context++);
	size_t count = stoi(tokens_[0]);
	EXPECT_EQ(true_count_context[context_string], count);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
