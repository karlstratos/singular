# Author: Karl Stratos (stratos@cs.columbia.edu)

# Compiler.
CC = clang++

# Warning level.
WARN = -Wall

# Optimization level.
OPT = -O3

# Where to find the SVDLIBC package.
SVDLIBC = third_party/SVDLIBC

# Where to find the Eigen package.
EIGEN = third_party/eigen-eigen-36fd1ba04c12

# Compiler flags
CFLAGS = $(WARN) $(OPT) -std=c++11
UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
# Tested on Apple (Darwin) clang version 4.0 (based on LLVM 3.1svn)
	CFLAGS += -stdlib=libc++
endif

# Extract object filenames by substituting ".cc" to ".o" in source filenames.
files = $(subst .cc,.o,$(shell ls *.cc) $(shell ls src/*.cc))

all: singular

singular: $(files) $(SVDLIBC)/libsvd.a
	$(CC) $(CFLAGS) $^ -o $@

%.o: %.cc
	$(CC) -I $(EIGEN) $(CFLAGS) -c $< -o $@

$(SVDLIBC)/libsvd.a:
	make -C $(SVDLIBC)

.PHONY: clean
clean:
	rm -rf *.o src/*.o singular
	make -C $(SVDLIBC) clean
