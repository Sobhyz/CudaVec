# CMake generated Testfile for 
# Source directory: /home/cip/ai2026/hy94qyxu/CudaVec
# Build directory: /home/cip/ai2026/hy94qyxu/CudaVec/build-release
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(VecDBTests "/home/cip/ai2026/hy94qyxu/CudaVec/build-release/test_vec_db")
set_tests_properties(VecDBTests PROPERTIES  _BACKTRACE_TRIPLES "/home/cip/ai2026/hy94qyxu/CudaVec/CMakeLists.txt;90;add_test;/home/cip/ai2026/hy94qyxu/CudaVec/CMakeLists.txt;0;")
add_test(SimilarityTests "/home/cip/ai2026/hy94qyxu/CudaVec/build-release/test_similarity")
set_tests_properties(SimilarityTests PROPERTIES  _BACKTRACE_TRIPLES "/home/cip/ai2026/hy94qyxu/CudaVec/CMakeLists.txt;97;add_test;/home/cip/ai2026/hy94qyxu/CudaVec/CMakeLists.txt;0;")
add_test(BenchmarkDatasetTests "/home/cip/ai2026/hy94qyxu/CudaVec/build-release/test_benchmark_dataset")
set_tests_properties(BenchmarkDatasetTests PROPERTIES  _BACKTRACE_TRIPLES "/home/cip/ai2026/hy94qyxu/CudaVec/CMakeLists.txt;103;add_test;/home/cip/ai2026/hy94qyxu/CudaVec/CMakeLists.txt;0;")
subdirs("_deps/json-build")
