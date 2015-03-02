# Author: Karl Stratos (stratos@cs.columbia.edu)

# Where to find the root of Google Test.
GTEST = gtest-1.7.0

# Where to find the SVDLIBC package.
SVDLIBC = ../third_party/SVDLIBC

# Where to find the Eigen package.
EIGEN = ../third_party/eigen-eigen-36fd1ba04c12

# C++ compiler.
CC = clang++

# Warning level.
WARN = -Wall

# Flags passed to the C++ compiler.
CFLAGS = -isystem $(GTEST)/include $(WARN) -std=c++11
UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
# Tested on Apple (Darwin) clang version 4.0 (based on LLVM 3.1svn)
	CFLAGS += -stdlib=libc++
endif

# Top-level commands.
TESTS = singular_test

all: $(TESTS)

.PHONY: clean
clean:
	rm -f $(TESTS) gtest.a gtest_main.a libsvd.a *.o ../src/*.o
	make -C $(SVDLIBC) clean

# All Google Test headers.  Usually you shouldn't change this
# definition.
GTEST_HEADERS = $(GTEST)/include/gtest/*.h \
                $(GTEST)/include/gtest/internal/*.h

# Builds gtest.a and gtest_main.a.

# Usually you shouldn't tweak such internal variables, indicated by a
# trailing _.
GTEST_SRCS_ = $(GTEST)/src/*.cc $(GTEST)/src/*.h $(GTEST_HEADERS)

# For simplicity and to avoid depending on Google Test's
# implementation details, the dependencies specified below are
# conservative and not optimized.  This is fine as Google Test
# compiles fast and for ordinary users its source rarely changes.
gtest-all.o: $(GTEST_SRCS_)
	     $(CC) -I $(GTEST) $(CFLAGS) -c $(GTEST)/src/gtest-all.cc

gtest_main.o: $(GTEST_SRCS_)
	      $(CC) -I $(GTEST) $(CFLAGS) -c $(GTEST)/src/gtest_main.cc

gtest.a: gtest-all.o
	 $(AR) $(ARFLAGS) $@ $^

gtest_main.a: gtest-all.o gtest_main.o
	      $(AR) $(ARFLAGS) $@ $^

# Builds tests.  A test should link with either gtest.a or
# gtest_main.a, depending on whether it defines its own main()
# function.

# Extract object filenames by substituting ".cc" to ".o" in source filenames.
files = $(subst .cc,.o,$(shell ls *.cc) $(shell ls ../src/*.cc))

singular_test: $(files) $(SVDLIBC)/libsvd.a gtest_main.a
	       $(CC) $(CFLAGS) $^ -o $@ -lpthread

%.o: %.cc
	$(CC) -I $(EIGEN) $(CFLAGS) -c $< -o $@

$(SVDLIBC)/libsvd.a:
	make -C $(SVDLIBC)
