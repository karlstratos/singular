# Author: Karl Stratos (stratos@cs.columbia.edu)
"""
This module is used to split a dataset into an held-out portion (odd lines) and
a final evaluation portion (even lines).
"""
import argparse

def main(args):
    """Splits data into dev and test portions."""
    odd_lines = []
    even_lines = []
    with open(args.data_path, "r") as data:
        line_num = 1
        for line in data:
            if line.split():  # Let's just keep non-empty lines.
                if line_num % 2:  # Even.
                    even_lines.append(line)
                else:  # Odd.
                    odd_lines.append(line)
                line_num += 1

    with open(args.output_dev_path, "w") as dev:
        for line in odd_lines:
            dev.write(line)

    with open(args.output_test_path, "w") as test:
        for line in even_lines:
            test.write(line)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data_path", type=str, help="path to data")
    argparser.add_argument("output_dev_path", type=str, help="path to output "
                           "dev set")
    argparser.add_argument("output_test_path", type=str, help="path to output "
                           "test set")
    parsed_args = argparser.parse_args()
    main(parsed_args)
