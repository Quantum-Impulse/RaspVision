# Define our host system
SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_VERSION 1)

# Define the cross compiler locations
SET(CMAKE_C_COMPILER /Users/erive/Downloads/raspbian10/bin/arm-raspbian10-linux-gnueabihf-gcc)
SET(CMAKE_CXX_COMPILER /Users/erive/Downloads/raspbian10/bin/arm-raspbian10-linux-gnueabihf-g++)

# Define the sysroot path for the RaspberryPi distribution in our tools folder 
SET(CMAKE_FIND_ROOT_PATH /Users/erive/Downloads/raspbian10/sys-root)

# Use our definitions for compiler tools
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# Search for libraries and headers in the target directories only
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
#add_compile_options(-Wall std=c++17 -Wl,--unresolved-symbols=ignore-in-shared-libs)
add_link_options(-Wl,--unresolved-symbols=ignore-in-shared-libs)

set(CMAKE_BUILD_TYPE Release)

#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

include_directories(BEFORE SYSTEM include include/opencv)
link_directories(lib)
# --unresolved-symbols=ignore-in-shared-libs

