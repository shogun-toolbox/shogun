set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(shogun_coreml_version 0.1.0)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/format")

set(HEADER_FILES ShogunCoreML.h CoreMLModel.h)
add_library(shogun-coreml SHARED
	CoreMLModel.cc
	internal/CoreMLConverter.cc
	internal/GLMRegressorConverter.cc
	internal/GLMClassifierConverter.cc
	internal/KernelConverter.cc
	internal/SVMClassifierConverter.cc
	internal/SVMRegressorConverter.cc)
set_property(TARGET shogun-coreml PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(
	shogun-coreml
	PROPERTIES
		SOVERSION 0
		VERSION "${shogun_coreml_version}"
		PUBLIC_HEADER "${HEADER_FILES}"
		ARCHIVE_OUTPUT_DIRECTORY "lib"
		LIBRARY_OUTPUT_DIRECTORY "lib"
)
target_link_libraries(shogun-coreml PUBLIC shogun::shogun ${Protobuf_LITE_LIBRARIES})
target_link_libraries(shogun-coreml PRIVATE coreml)
target_include_directories(shogun-coreml PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:include/shogun-coreml>
)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/internal")

set(SHOGUN_COREML_LIB_INSTALL lib)
file(TO_CMAKE_PATH ${SHOGUN_COREML_LIB_INSTALL}/cmake/shogun-coreml CONFIG_PACKAGE_DIR)

install(
	TARGETS shogun-coreml
	EXPORT ShogunCoreMLTarget
	ARCHIVE DESTINATION ${SHOGUN_COREML_LIB_INSTALL}
	RUNTIME DESTINATION ${SHOGUN_COREML_LIB_INSTALL}
	LIBRARY DESTINATION ${SHOGUN_COREML_LIB_INSTALL}
	INCLUDES DESTINATION include/shogun-coreml
	PUBLIC_HEADER DESTINATION include/shogun-coreml
)

export(EXPORT ShogunCoreMLTarget
  FILE "${CMAKE_CURRENT_BINARY_DIR}/ShogunCoreMLTarget.cmake"
  NAMESPACE shogun::
)

install(EXPORT ShogunCoreMLTarget
  FILE ShogunCoreMLTarget.cmake
  NAMESPACE shogun::
  DESTINATION ${CONFIG_PACKAGE_DIR}
)
