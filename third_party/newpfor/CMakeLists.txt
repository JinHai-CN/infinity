project(NewPFOR CXX C)

# library target
add_library(newpfor STATIC
    decompress_sse4.cpp
    new_pfordelta_compressor.cpp)
set_target_properties(newpfor PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
