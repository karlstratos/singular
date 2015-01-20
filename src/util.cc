// Author: Karl Stratos (karlstratos@gmail.com)

#include "util.h"

#include <algorithm>
#include <math.h>
#include <random>
#include <sys/stat.h>
#include <unordered_map>

void StringManipulator::split(const string &line, const string &delimiter,
			      vector<string> *tokens) {
    tokens->clear();
    size_t start = 0;  // Keep track of the current position.
    size_t end = 0;
    string token;
    while (end != string::npos) {
	end = line.find(delimiter, start);  // Where the delimiter occurs next.

	// Collect a corresponding portion of the line into a token.
	token = (end == string::npos) ?
	    line.substr(start, string::npos) : line.substr(start, end - start);
	if(token != "") { tokens->push_back(token); }

	// Update the current position.
	start = (end > string::npos - delimiter.size()) ?
	    string::npos : end + delimiter.size();
    }
}

string StringManipulator::print_time(double num_seconds) {
    int num_hours = (int) floor(num_seconds / 3600.0);
    int h_seconds = num_hours * 3600;
    double num_seconds_minus_h = (double) (num_seconds - h_seconds);
    int num_minutes = (int) floor(num_seconds_minus_h / 60.0);
    int m_seconds = num_minutes * 60;
    int num_seconds_minus_hm = num_seconds_minus_h - m_seconds;
    string time_string = to_string(num_hours) + "h" + to_string(num_minutes)
	+ "m" + to_string(num_seconds_minus_hm) + "s";
    return time_string;
}

bool FileManipulator::exists(const string &file_path) {
    struct stat buffer;
    return (stat(file_path.c_str(), &buffer) == 0);
}

double Stat::ComputeSpearman(const vector<double> &x, const vector<double> &y) {
    ASSERT(x.size() == y.size(), "Given two vectors of different lengths: "
	   << x.size() << " " << y.size());
    double tol = 1e-10;

    // Variable x: sort values and compute tie-averaged ranks.
    vector<double> x_sorted = x;
    sort(x_sorted.begin(), x_sorted.end());
    vector<double> x_averaged_ranks;  // List of tie-averaged ranks.
    size_t while_index = 0;
    while (while_index < x_sorted.size()) {
	size_t num_ties = 1;
	size_t sum_rank = while_index + 1;
	while (while_index + 1 < x_sorted.size() &&
	       fabs(x_sorted[while_index + 1] - x_sorted[while_index]) < tol) {
	    ++while_index;
	    ++num_ties;
	    sum_rank += while_index + 1;
	}
	for (size_t j = 0; j < num_ties; ++j) {
	    // Assign the average rank to all tied elements.
	    x_averaged_ranks.push_back(((double) sum_rank) / num_ties);
	}
	++while_index;
    }

    // Sort values in variable y and record their sorted indices.
    vector<double> y_sorted = y;
    sort(y_sorted.begin(), y_sorted.end());
    vector<double> y_averaged_ranks;  // List of tie-averaged ranks.
    while_index = 0;
    while (while_index < y_sorted.size()) {
	size_t num_ties = 1;
	size_t sum_rank = while_index + 1;
	while (while_index + 1 < y_sorted.size() &&
	       fabs(y_sorted[while_index + 1] - y_sorted[while_index]) < tol) {
	    ++while_index;
	    ++num_ties;
	    sum_rank += while_index + 1;
	}
	for (size_t j = 0; j < num_ties; ++j) {
	    // Assign the average rank to all tied elements.
	    y_averaged_ranks.push_back(((double) sum_rank) / num_ties);
	}
	++while_index;
    }

    // Prepare a final pair of tie-averaged ranks.
    unordered_map<double, size_t> x_val2ind;
    for (size_t index = 0; index < x_sorted.size(); ++index) {
	// A bit of cheating:
	// Note that if x has same values, this mapping may have duplicate,
	// overwritten indices. But that's fine since any of such indices will
	// point to the same tie-averaged rank.
	x_val2ind[x_sorted[index]] = index;
    }
    vector<double> x_spearman;
    for (double val : x) {
	size_t index = x_val2ind[val];
	x_spearman.push_back(x_averaged_ranks[index]);
    }
    unordered_map<double, size_t> y_val2ind;
    for (size_t index = 0; index < y_sorted.size(); ++index) {
	// A bit of cheating:
	// Note that if y has same values, this mapping may have duplicate,
	// overwritten indices. But that's fine since any of such indices will
	// point to the same tie-averaged rank.
	y_val2ind[y_sorted[index]] = index;
    }
    vector<double> y_spearman;
    for (double val : y) {
	size_t index = y_val2ind[val];
	y_spearman.push_back(y_averaged_ranks[index]);
    }

    double sum_squared_diff = 0;
    for (size_t i = 0; i < x_spearman.size(); ++i) {
	sum_squared_diff += pow(x_spearman[i] - y_spearman[i], 2);
    }
    double spearman_corr = 1.0 - (6.0 * sum_squared_diff) /
	(x.size() * (pow(x.size(), 2) - 1));

    return spearman_corr;
}

// [*Warning* Unchecked for correctness.]
void Sampler::sample_indices_without_replacement(size_t range_cap,
						 size_t num_samples,
						 vector<size_t> *samples) {
    size_t num_addressed = 0;  // Number of indices addressed so far.
    size_t num_selected = 0;  // Number of indices actually selected so far.
    (*samples).resize(num_samples);

    while (num_selected < num_samples) {
	random_device device;
	default_random_engine engine(device());
	uniform_real_distribution<double> unif(0.0, 1.0);
	double u = unif(engine);  // u from (0, 1) uniformly at random.

        if ( (range_cap - num_addressed) * u < num_samples - num_selected ) {
            (*samples)[num_selected++] = num_addressed;
        }
	++num_addressed;
    }
}
