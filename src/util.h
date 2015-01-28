// Author: Karl Stratos (karlstratos@gmail.com)
//
// Various utility functions and classes.

#ifndef UTIL_H_
#define UTIL_H_

#include <iostream>
#include <string>
#include <vector>

using namespace std;

class StringManipulator {
public:
    // Splits the given line by delimiter into tokens.
    void split(const string &line, const string &delimiter,
	       vector<string> *tokens);

    // Q seconds => "XhYmZs" (X hours Y minutes Z seconds)
    string print_time(double num_seconds);

    // Returns the lowercase transformation of the given string.
    string lowercase(const string &original_string);
};

class FileManipulator {
public:
    // Checks for the existence of a file.
    bool exists(const string &file_path);
};

class Stat {
public:
    // Computes Spearman's rank correlation coefficient.
    double ComputeSpearman(const vector<double> &x, const vector<double> &y);
};

class Sampler {
public:
    // [*Warning* Unchecked for correctness.]
    // Samples the specified number of indices from a given range [0, range_cap)
    // without replacement.
    void sample_indices_without_replacement(size_t range_cap,
					    size_t num_samples,
					    vector<size_t> *samples);
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

#endif  // UTIL_H_
