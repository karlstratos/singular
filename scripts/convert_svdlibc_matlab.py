# Author: Karl Stratos (stratos@cs.columbia.edu)
"""
This module is used to convert a sparse matrix from the SVDLIBC format (i.e.,
column-major) to the Matlab format ({row col val}).

Argument 1 [input: SVDLIBC sparse matrix text file]
Argument 2 [output: Matlab sparse matrix text file]
"""
import sys

def convert_svdlibc_matlab(svdlibc_path, matlab_path):
    """
    Convert a sparse matrix from the SVDLIBC format to the Matlab format.
    """
    # Read a sparse matrix in the SVDLIBC format.
    matrix = {}
    with open(svdlibc_path, "r") as svdlibc_file:
        tokens = svdlibc_file.readline().split()
        num_columns = int(tokens[1])
        for col in range(num_columns):
            num_nonzero_rows = int(svdlibc_file.readline().split()[0])
            for _ in range(num_nonzero_rows):
                tokens = svdlibc_file.readline().split()
                row = int(tokens[0])
                value = float(tokens[1])
                matrix[(row, col)] = value

    # Write it in Matlab format.
    with open(matlab_path, "w") as matlab_file:
        for (row, col) in matrix:
            matlab_file.write(str(row + 1) + " " + str(col + 1) + " " +
                              str(matrix[(row, col)]) + "\n")

if __name__ == "__main__":
    # Path to input.
    SVDLIBC_PATH = sys.argv[1]

    # Path to output.
    MATLAB_PATH = sys.argv[2]

    convert_svdlibc_matlab(SVDLIBC_PATH, MATLAB_PATH)
