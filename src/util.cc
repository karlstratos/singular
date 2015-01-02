// Author: Karl Stratos (karlstratos@gmail.com)

#include "util.h"

#include <math.h>
#include <random>
#include <sys/stat.h>

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
