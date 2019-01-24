#include "internal/CoreMLConverter.h"
#include "../ShogunCoreML.h"

#include <shogun/base/Version.h>
#include <shogun/lib/exception/NotFittedException.h>

using namespace shogun;
using namespace shogun::coreml;

std::shared_ptr<CoreMLModel> shogun::coreml::convert(const CMachine* m) noexcept(false)
{
	REQUIRE(m != NULL, "No machine has been provided")

	auto converter_registry = ConverterFactory::instance();
	if (!m->is_trained())
		throw NotFittedException("The supplied machine is not trained!");

	auto spec = converter_registry->operator()(m)->description();
	spec->mutable_description()->mutable_metadata()->set_versionstring(Version::get_version_main());
	return std::make_shared<CoreMLModel>(spec);
}
