#include "base/SGObject.h"
#include "lib/io.h"
#include "base/Parallel.h"
#include "base/Version.h"

#ifndef HAVE_SWIG
CParallel CSGObject::parallel;
CIO CSGObject::io;
CVersion CSGObjectversion;
#endif
