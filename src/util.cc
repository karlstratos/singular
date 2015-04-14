// Author: Karl Stratos (stratos@cs.columbia.edu)

#include "util.h"

#include <algorithm>
#include <dirent.h>
#include <math.h>
#include <random>
#include <sys/stat.h>
#include <unordered_map>

void StringManipulator::Split(const string &line, const string &delimiter,
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

string StringManipulator::TimeString(double num_seconds) {
    size_t num_hours = (int) floor(num_seconds / 3600.0);
    double num_seconds_minus_h = num_seconds - (num_hours * 3600);
    int num_minutes = (int) floor(num_seconds_minus_h / 60.0);
    int num_seconds_minus_hm = num_seconds_minus_h - (num_minutes * 60);
    string time_string = to_string(num_hours) + "h" + to_string(num_minutes)
	+ "m" + to_string(num_seconds_minus_hm) + "s";
    return time_string;
}

string StringManipulator::Lowercase(const string &original_string) {
    string lowercased_string;
    for (const char &character : original_string) {
	lowercased_string.push_back(tolower(character));
    }
    return lowercased_string;
}

bool FileManipulator::Exists(const string &file_path) {
    struct stat buffer;
    return (stat(file_path.c_str(), &buffer) == 0);
}

string FileManipulator::FileType(const string &file_path) {
    string file_type;
    struct stat stat_buffer;
    if (stat(file_path.c_str(), &stat_buffer) == 0) {
	if (stat_buffer.st_mode & S_IFREG) {
	    file_type = "file";
	} else if (stat_buffer.st_mode & S_IFDIR) {
	    file_type = "dir";
	} else {
	    file_type = "other";
	}
    } else {
	ASSERT(false, "Problem with " << file_path);
    }
    return file_type;
}

void FileManipulator::ListFiles(const string &file_path, vector<string> *list) {
    (*list).clear();
    string file_type = FileType(file_path);
    if (file_type == "dir") {
	DIR *pDIR = opendir(file_path.c_str());
	if (pDIR != NULL) {
	    struct dirent *entry = readdir(pDIR);
	    while (entry != NULL) {
		if (strcmp(entry->d_name, ".") != 0 &&
		    strcmp(entry->d_name, "..") != 0) {
		    (*list).push_back(file_path + "/" + entry->d_name);
		}
		entry = readdir(pDIR);
	    }
        }
	closedir(pDIR);
    } else {
	(*list).push_back(file_path);
    }
}

size_t FileManipulator::NumLines(const string &file_path) {
    size_t num_lines = 0;
    ifstream file(file_path, ios::in);
    ASSERT(file.is_open(), "Cannot open file: " << file_path);
    string line;
    while (getline(file, line)) { ++num_lines; }
    return num_lines;
}

void FileManipulator::Write(const Eigen::MatrixXd &m, const string &file_path) {
    ofstream file(file_path, ios::out);
    ASSERT(file.is_open(), "Cannot open file: " << file_path);
    file << m.rows() << " " << m.cols() << endl;
    for (size_t i = 0; i < m.rows(); ++i) {
	for (size_t j = 0; j < m.cols(); ++j) {
	    file << m(i, j) << endl;
	}
    }
}

void FileManipulator::Write(const Eigen::VectorXd &v, const string &file_path) {
    ofstream file(file_path, ios::out);
    ASSERT(file.is_open(), "Cannot open file: " << file_path);
    for (size_t i = 0; i < v.size(); ++i) {
	file << v(i) << endl;
    }
}

void FileManipulator::Read(const string &file_path, Eigen::MatrixXd *m) {
    m->resize(0, 0);  // Clear the m.
    ifstream file(file_path, ios::in);
    ASSERT(file.is_open(), "Cannot open file: " << file_path);
    StringManipulator string_manipulator;
    string line;
    vector<string> tokens;

    // Get dimensions.
    getline(file, line);
    string_manipulator.Split(line, " ", &tokens);
    ASSERT(tokens.size() == 2, "Bad m format: " << line);
    size_t dim1 = stol(tokens[0]);
    size_t dim2 = stol(tokens[1]);

    // Get entries.
    m->resize(dim1, dim2);
    for (size_t i = 0; i < dim1; ++i) {
	for (size_t j = 0; j < dim2; ++j) {
	    getline(file, line);
	    string_manipulator.Split(line, " ", &tokens);
	    ASSERT(tokens.size() == 1, "Bad format: " << line);
	    (*m)(i, j) = stod(tokens[0]);
	}
    }
}

void FileManipulator::Read(const string &file_path, Eigen::VectorXd *v) {
    v->resize(0);  // Clear the vector.

    // Get values as a map.
    unordered_map<size_t, double> values;
    Read(file_path, &values);

    // Fill the vector.
    v->resize(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
	(*v)(i) = values[i];
    }
}

void FileManipulator::Read(const string &values_path,
			   unordered_map<size_t, double> *values) {
    ifstream file(values_path, ios::in);
    ASSERT(file.is_open(), "Cannot open file: " << values_path);
    StringManipulator string_manipulator;
    string line;
    vector<string> tokens;
    values->clear();
    size_t index = 0;
    while (file.good()) {
	getline(file, line);  // Scaling value for the i-th dimension.
	if (line == "") { continue; }
	string_manipulator.Split(line, " ", &tokens);
	ASSERT(tokens.size() == 1, "Bad format: " << line);
	(*values)[index++] = stod(tokens[0]);
    }
}

Eigen::MatrixXd LinearAlgebra::ComputePinv(const Eigen::MatrixXd &M) {
    double tol = 1e-6;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU |
					  Eigen::ComputeThinV);
    Eigen::VectorXd inverse_singular_values(svd.singularValues().size());
    for (size_t i = 0; i < svd.singularValues().size(); ++i) {
	if (svd.singularValues()(i) > tol) {
	    inverse_singular_values(i) = 1.0 / svd.singularValues()(i);
	} else {
	    inverse_singular_values(i) = 0.0;
	}
    }
    return svd.matrixV() * inverse_singular_values.asDiagonal() *
	svd.matrixU().transpose();
}

double Stat::ComputeSpearman(const vector<double> &values1,
			     const vector<double> &values2) {
    ASSERT(values1.size() == values2.size(), "Different lengths: "
	   << values1.size() << " " << values2.size());

    vector<double> values1_transformed;
    vector<double> values2_transformed;
    AverageRankTransform(values1, &values1_transformed);
    AverageRankTransform(values2, &values2_transformed);

    size_t num_instances = values1.size();
    double sum_squares = 0;
    for (size_t i = 0; i < num_instances; ++i) {
	sum_squares += pow(values1_transformed[i] - values2_transformed[i], 2);
    }
    double uncorrelatedness = 6.0 * sum_squares /
	(num_instances * (pow(num_instances, 2) - 1));
    double corrleation = 1.0 - uncorrelatedness;
    return corrleation;
}

void Stat::AverageRankTransform(const vector<double> &values,
				vector<double> *transformed_values) {
    transformed_values->clear();
    vector<double> sorted_values = values;
    sort(sorted_values.begin(), sorted_values.end());
    vector<double> averaged_ranks;
    size_t index = 0;
    while (index < sorted_values.size()) {
	size_t num_same = 1;
	size_t rank_sum = index + 1;
	while (index + 1 < sorted_values.size() &&
	       fabs(sorted_values[index + 1] - sorted_values[index]) < 1e-15) {
	    ++index;
	    ++num_same;
	    rank_sum += index + 1;
	}

	double averaged_rank = ((double) rank_sum) / num_same;
	for (size_t j = 0; j < num_same; ++j) {
	    // Assign the average rank to all tied elements.
	    averaged_ranks.push_back(averaged_rank);
	}
	++index;
    }

    // Map each value to the corresponding index in averaged_ranks. A value can
    // appear many times but it doesn't matter since it will have the same
    // averaged rank.
    unordered_map<double, size_t> value2index;
    for (size_t index = 0; index < sorted_values.size(); ++index) {
	value2index[sorted_values[index]] = index;
    }
    for (double value : values) {
	size_t index = value2index[value];
	transformed_values->push_back(averaged_ranks[index]);
    }
}
