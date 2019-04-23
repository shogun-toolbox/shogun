#include <shogun/features/DummyFeatures.h>
#include <shogun/base/Parameter.h>

using namespace shogun;

DummyFeatures::DummyFeatures()
{
	init();
	num_vectors = 0;
}

DummyFeatures::DummyFeatures(int32_t num) : Features(0), num_vectors(num)
{
	init();
}

DummyFeatures::DummyFeatures(const DummyFeatures &orig) : Features(0),
	num_vectors(orig.num_vectors)
{
	init();
}

DummyFeatures::~DummyFeatures()
{
}

int32_t DummyFeatures::get_num_vectors() const
{
	return num_vectors;
}

std::shared_ptr<Features> DummyFeatures::duplicate() const
{
	return std::make_shared<DummyFeatures>(*this);
}

inline EFeatureType DummyFeatures::get_feature_type() const
{
	return F_ANY;
}

EFeatureClass DummyFeatures::get_feature_class() const
{
	return C_ANY;
}

void DummyFeatures::init()
{
	SG_ADD(
	    &num_vectors, "num_vectors", "Number of feature vectors.");
}
