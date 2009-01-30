#include "base/SGObject.h"
#include "lib/io.h"
#include "lib/Mathematics.h"
#include "base/Parallel.h"
#include "base/Version.h"

CParallel CSGObject::parallel;
CIO CSGObject::io;
CVersion CSGObject::version;
CIO* sg_io=&CSGObject::io;

//this creates a math object for the purpose of the constructor to be called at least once
volatile CMath math;
