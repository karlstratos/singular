// Author: Karl Stratos (karlstratos@gmail.com)
//
// Various utility functions and classes.

#ifndef UTIL_H_
#define UTIL_H_

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
};

// Assert macro that allows adding a message to an assertion upon failure. It
// implictly performs string conversion: ASSERT(x > 0, "Negative x: " << x);
#ifndef NDEBUG
#   define ASSERT(condition, message) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << message << std::endl; \
            std::exit(EXIT_FAILURE); \
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
