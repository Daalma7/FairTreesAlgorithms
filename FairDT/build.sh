#!/bin/bash
# Build c files from pyx and pxd files
cython _utils.pyx
cython _tree.pyx
cython _criterion.pyx
cython _splitter.pyx

# We build the so files using this command. They end up in a build folder
python setup.py build_ext --inplace

# Copy those files to the main package folder
yes | cp -rf build/lib.macosx-10.9-x86_64-3.9/FairDT/* .

# Remove the build folder
rm -r build

# Also clean c files
rm *.c


# Interesting links
#https://stackoverflow.com/questions/40033222/edit-scikit-learn-decisiontree
#https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09bcc2eaeba98f7e737aac2ac782f0e5f1/sklearn/#tree/_criterion.pyx