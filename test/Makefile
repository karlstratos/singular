# Author: Karl Stratos (karlstratos@gmail.com)

# Points to the root of Google Test.
GTEST_DIR = gtest-1.7.0

# Where to find user source code.
USER_DIR = ..

# Where to find user test code.
TEST_DIR = .

# Where to find the Eigen package.
EIGEN = ../third_party/eigen-eigen-36fd1ba04c12

# Flags passed to the preprocessor.
# Set Google Test's header directory as a system directory, such that
# the compiler doesn't generate warnings in Google Test headers.
CPPFLAGS += -isystem $(GTEST_DIR)/include

# C++ compiler.
CXX = clang++

# Warning level.
WARN = -Wall

# Debug information
DEBUG = -g

# Flags passed to the C++ compiler.
CXXFLAGS = $(WARN) $(DEBUG) -std=c++11
UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
# Tested on Apple (Darwin) clang version 4.0 (based on LLVM 3.1svn)
	CXXFLAGS += -stdlib=libc++
endif

# All tests produced by this Makefile.  Remember to add new tests you
# created to the list.
TESTS = $(SINGULAR)

SINGULAR = singular_test

# All Google Test headers.  Usually you shouldn't change this
# definition.
GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
                $(GTEST_DIR)/include/gtest/internal/*.h

# House-keeping build targets.

all: $(TESTS)

clean:
	rm -f $(TESTS) gtest.a gtest_main.a libsvd.a *.o
	make -C $(USER_DIR)/third_party/SVDLIBC clean

# Builds gtest.a and gtest_main.a.

# Usually you shouldn't tweak such internal variables, indicated by a
# trailing _.
GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)

# For simplicity and to avoid depending on Google Test's
# implementation details, the dependencies specified below are
# conservative and not optimized.  This is fine as Google Test
# compiles fast and for ordinary users its source rarely changes.
gtest-all.o: $(GTEST_SRCS_)
	     $(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
	     $(GTEST_DIR)/src/gtest-all.cc

gtest_main.o: $(GTEST_SRCS_)
	      $(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c \
              $(GTEST_DIR)/src/gtest_main.cc

gtest.a: gtest-all.o
	 $(AR) $(ARFLAGS) $@ $^

gtest_main.a: gtest-all.o gtest_main.o
	      $(AR) $(ARFLAGS) $@ $^

# Builds tests.  A test should link with either gtest.a or
# gtest_main.a, depending on whether it defines its own main()
# function.

# Test files in the source directory.
singular_test: singular_test.o decompose.o sparsesvd.o cluster.o util.o \
	       gtest_main.a wordrep.o libsvd.a
	       $(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@ -lpthread

singular_test.o: decompose.o sparsesvd.o
	         $(CXX) -I $(EIGEN) $(CPPFLAGS) $(CXXFLAGS) -c \
	         $(TEST_DIR)/singular_test.cc

wordrep.o: decompose.o cluster.o
	   $(CXX) -I $(EIGEN) $(CXXFLAGS) -c $(USER_DIR)/src/wordrep.cc

decompose.o: sparsesvd.o
	     $(CXX) -I $(EIGEN) $(CXXFLAGS) -c $(USER_DIR)/src/decompose.cc

sparsesvd.o: util.o libsvd.a
	     $(CXX) $(CXXFLAGS) -c $(USER_DIR)/src/sparsesvd.cc

cluster.o: util.o
	$(CXX) -I $(EIGEN) $(CXXFLAGS) -c $(USER_DIR)/src/cluster.cc

util.o:
	$(CXX) $(CXXFLAGS) -c $(USER_DIR)/src/util.cc

libsvd.a:
	make -C $(USER_DIR)/third_party/SVDLIBC
	cp $(USER_DIR)/third_party/SVDLIBC/libsvd.a .
