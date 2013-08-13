# Copyright (C) 2007-2012 Hypertable, Inc.
#
# This file is part of Hypertable.
#
# Hypertable is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or any later version.
#
# Hypertable is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Hypertable. If not, see <http://www.gnu.org/licenses/>
#

# - Find Hoard
# Find the native Hoard includes and library
#
# Hoard_LIBRARIES - List of libraries when using Hoard.
# Hoard_FOUND - True if Hoard found.

set(Hoard_NAMES hoard)

find_library(Hoard_LIBRARY NO_DEFAULT_PATH
  NAMES ${Hoard_NAMES}
  PATHS ${HT_DEPENDENCY_LIB_DIR} /lib /usr/lib /usr/local/lib /opt/local/lib
)

if (Hoard_LIBRARY)
  set(Hoard_FOUND TRUE)
  set( Hoard_LIBRARIES ${Hoard_LIBRARY} )
else ()
  set(Hoard_FOUND FALSE)
  set( Hoard_LIBRARIES )
endif ()

if (Hoard_FOUND)
  message(STATUS "Found Hoard: ${Hoard_LIBRARY}")
else ()
  message(STATUS "Not Found Hoard: ${Hoard_LIBRARY}")
endif ()

mark_as_advanced(
  Hoard_LIBRARIES
)
