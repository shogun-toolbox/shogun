set(STAN_STABLE_RELEASE 4b1a10bc877d941bbe0a12c198526807be27167a)

set(STAN_INCLUDE_DIR_STAN_MATH ${CMAKE_BINARY_DIR}/StanMath/src/StanMath)
set(STAN_INCLUDE_DIR_BOOST ${CMAKE_BINARY_DIR}/StanMath/src/StanMath/lib/boost_1.64.0)
set(STAN_INCLUDE_DIR_CVODES ${CMAKE_BINARY_DIR}/StanMath/src/StanMath/lib/cvodes_2.9.0/include)

include(ExternalProject)
ExternalProject_Add(
	StanMath
	PREFIX ${CMAKE_BINARY_DIR}/StanMath
	DOWNLOAD_DIR ${THIRD_PARTY_DIR}/StanMath
	GIT_REPOSITORY https://github.com/stan-dev/math
	GIT_TAG ${STAN_STABLE_RELEASE}
	CONFIGURE_COMMAND ""
	BUILD_COMMAND ""
	INSTALL_COMMAND
		${CMAKE_COMMAND} -E copy_if_different ${STAN_INCLUDE_DIR_STAN_MATH} include/shogun/third_party/Stan
		&& ${CMAKE_COMMAND} -E copy_if_different ${STAN_INCLUDE_DIR_BOOST} include/shogun/third_party/Stan_Boost
		&& ${CMAKE_COMMAND} -E copy_if_different ${STAN_INCLUDE_DIR_CVODES} include/shogun/third_party/Stan_Cvodes
	LOG_DOWNLOAD ON
	)
add_dependencies(libshogun StanMath)
