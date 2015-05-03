// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Various utility functions and classes.

#ifndef UTIL_H_
#define UTIL_H_

#include <Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

// Class for manipulating strings.
class StringManipulator {
public:
    // Splits the given line by delimiter into tokens.
    void Split(const string &line, const string &delimiter,
	       vector<string> *tokens);

    // Returns the hour/minute/second string of seconds: 6666 => "1h51m6s".
    string TimeString(double num_seconds);

    // Returns the lowercase transformation of the given string.
    string Lowercase(const string &original_string);

    // Returns the string form of a double.
    string DoubleString(double value, size_t decimal_place,
			bool replace_period);
};

// Class for manipulating files.
class FileManipulator {
public:
    // Returns true if the file exists, false otherwise.
    bool Exists(const string &file_path);

    // Returns the type of the given file path: "file", "dir", or "other".
    string FileType(const string &file_path);

    // List files. If given a single file, the list contains the path to that
    // file. If given a directory, the list contains the paths to the files
    // inside that directory (non-recursively).
    void ListFiles(const string &file_path, vector<string> *list);

    // Returns the number of lines in a file.
    size_t NumLines(const string &file_path);

    // Writes an Eigen matrix to a text file.
    void Write(const Eigen::MatrixXd &m, const string &file_path);

    // Writes an Eigen vector to a text file.
    void Write(const Eigen::VectorXd &v, const string &file_path);

    // Reads an Eigen matrix from a text file.
    void Read(const string &file_path, Eigen::MatrixXd *m);

    // Reads an Eigen vector from a text file.
    void Read(const string &file_path, Eigen::VectorXd *v);

    // Reads an index:value map from lines of values.
    void Read(const string &values_path, unordered_map<size_t, double> *values);
};

// Class for linear algebraic operations not already supported.
class LinearAlgebra {
public:
    // Computes the Moore–Penrose pseudo-inverse of a given matrix.
    Eigen::MatrixXd ComputePinv(const Eigen::MatrixXd &M);
};

class Stat {
public:
    // Computes Spearman's rank correlation coefficient between two sequences
    // of values.
    double ComputeSpearman(const vector<double> &values1,
			   const vector<double> &values2);

    // Computes the average-rank transformation of a sequence of values.
    void AverageRankTransform(const vector<double> &values,
			      vector<double> *transformed_values);
};

// Assert macro that allows adding a message to an assertion upon failure. It
// implictly performs string conversion: ASSERT(x > 0, "Negative x: " << x);
#ifndef NDEBUG
#   define ASSERT(condition, message) \
    do { \
        if (! (condition)) { \
            cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << message << endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (false)
#else
#   define ASSERT(condition, message) do { } while (false)
#endif

// Template for a struct used to sort a vector of pairs by value. Use it like:
//    sort(v.begin(), v.end(), sort_pairs_second<int, int>());
//    sort(v.begin(), v.end(), sort_pairs_second<int, int, greater<int> >());
template <class T1, class T2, class Predicate = less<T2> >
struct sort_pairs_second {
    bool operator()(const pair<T1, T2> &left, const pair<T1, T2> &right) {
        return Predicate()(left.second, right.second);
    }
};

// Template for string conversion of floating points with precision.
template <typename T>
string to_string_with_precision(const T value, const int precision = 2) {
    ostringstream out;
    out << setprecision(precision) << value;
    return out.str();
}

#endif  // UTIL_H_
