
SET(_bindir "${CMAKE_MODULE_PATH}/")
TRY_COMPILE(HAVE_SPINLOCK "${_bindir}" "${CMAKE_MODULE_PATH}/spinlock-test.cpp")

if (HAVE_SPINLOCK)
	MESSAGE(STATUS "Spinlock support found")
	SET(SPINLOCK_FOUND TRUE)
else (HAVE_SPINLOCK)
	MESSAGE(STATUS "Spinlock support not found")
	SET(SPINLOCK_FOUND FALSE)
endif (HAVE_SPINLOCK)
