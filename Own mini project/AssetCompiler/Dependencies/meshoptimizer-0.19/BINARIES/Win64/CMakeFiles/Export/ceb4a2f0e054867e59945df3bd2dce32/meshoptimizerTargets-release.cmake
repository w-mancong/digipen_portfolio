#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "meshoptimizer::meshoptimizer" for configuration "Release"
set_property(TARGET meshoptimizer::meshoptimizer APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(meshoptimizer::meshoptimizer PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/meshoptimizer.lib"
  )

list(APPEND _cmake_import_check_targets meshoptimizer::meshoptimizer )
list(APPEND _cmake_import_check_files_for_meshoptimizer::meshoptimizer "${_IMPORT_PREFIX}/lib/meshoptimizer.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
