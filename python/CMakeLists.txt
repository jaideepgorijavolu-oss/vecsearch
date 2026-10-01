find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core MODULE bindings.cpp)
target_link_libraries(_core PRIVATE vecsearch)
install(TARGETS _core LIBRARY DESTINATION vecsearch)
