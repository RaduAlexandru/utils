# utils

This is a loose collection of C++ 14 utilities that I tend to use between various projects. 

### Usage 
Most utilities are self contained and can be built separely. 
Some are header-only which means only including the relevant `.h` file.
To use a single library, just `#include` the header, and if there is a `.cpp` either compile and link that, or just include that too, e.g.:

	#include <utils/ColormapMngr.h>
	#include <utils/ColormapMngr.cxx> // Do this only in one of your .cpp files

You also need to compile Loguru. Luckily, `loguru.hpp` is included in this repo, and to compile it you just need to add the following to one of your .cpp files:

	#define LOGURU_IMPLEMENTATION
	#include <loguru.hpp>

Make sure you compile with -std=c++11 -lpthread -ldl 

If you have CMake >=3.14 you can obtain all the utils by adding to your CMakeLists.txt:

    #Clones the utils
    FetchContent_Declare(
    utils
    GIT_REPOSITORY https://github.com/RaduAlexandru/utils.git )
    FetchContent_MakeAvailable(utils)
    
    include_directories(${utils_SOURCE_DIR}/include)
    target_link_libraries(<PROJECT_NAME> utils )

If you have an older version of CMake (<3.14) you can also add the utilities as a git submodule:

    git submodule add https://github.com/RaduAlexandru/utils.git extern/utils
and compiling and linking in your CMakeLists.txt: 

    add_subdirectory (utils)
    target_link_libraries(<PROJECT_NAME> utils )
You can also compile all the utilities at once by adding `add_subdirectory (utils)` to your CMakeLists.txt This creates the `utils` library which can be linked using `target_link_libraries(<PROJECT_NAME> utils )`

#### Documentation
This file (README.md) contains an overview of each library. Read the header for each library to learn more.

#### Tests
There is a very limited set of tests in the `tests/` folder.

# Stand-alone libraries

#### profiler.h

#### eigen_utils.h
Useful functions to have for manipulating Eigen matrices (removing rows, columns, reindexing, etc). They are particularly useful for processing meshes which are represented as matrices of vertices V and faces F. Easy_PBR heavily uses this library for aiding geometry processing. 

#### string_utils.h 
    trimming 
    and formatting

#### opencv_utils.h 
    adding alpha 
    easy query and display the type of a matrix in a human understandable way 
    
#### RandGenerator.h
    random number generator 
    
#### numerical_utils.h
    interpolation and shaping functions functions (step, smoothstep)
    indexing utilities (3D indices xyz -> linear index )
    number manipulation ( degree2radians, clamping etc)
   
#### Ringbuffer.h 

#### ColorMngr.h