IF (ENABLE_TESTING)
	include(external/GoogleTestNMock)
	ExternalProject_Get_Property(GoogleMock source_dir)
	LINK_DIRECTORIES(${THIRD_PARTY_DIR}/libs/gmock)

	enable_testing()
	add_library(coreml_unittest_main ${CMAKE_CURRENT_SOURCE_DIR}/unittest_main.cc)
	target_link_libraries(coreml_unittest_main gmock gtest shogun::shogun)
	add_dependencies(coreml_unittest_main GoogleMock shogun::shogun)
	target_include_directories(coreml_unittest_main
		PRIVATE
		${source_dir}/googlemock/include
		${source_dir}/googletest/include)
	set(SHOGUN_UNITTEST_LINK_LIBS coreml_unittest_main shogun::shogun shogun-coreml gmock gtest)

	ADD_SHOGUN_UNITTEST(TARGET SVMRegressorConverter_test LABELS coreml-unit)
	ADD_SHOGUN_UNITTEST(TARGET SVMClassifierConverter_test LABELS coreml-unit)
	ADD_SHOGUN_UNITTEST(TARGET KernelConverter_test LABELS coreml-unit)
	ADD_SHOGUN_UNITTEST(TARGET GLMRegressorConverter_test LABELS coreml-unit)
	ADD_SHOGUN_UNITTEST(TARGET GLMClassifierConverter_test LABELS coreml-unit)
ENDIF()
