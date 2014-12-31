// Author: Karl Stratos (karlstratos@gmail.com)

#include "util.h"

#include <math.h>
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
