#ifndef SHOGUN_FMTLIB_H
#define SHOGUN_FMTLIB_H

#if defined(USE_EXTERNAL_SPDLOG)
#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/bundled/ostream.h>
#else
#include <shogun/third_party/spdlog/fmt/fmt.h>
#include <shogun/third_party/spdlog/fmt/bundled/ostream.h>
#endif

#endif // SHOGUN_FMTLIB_H