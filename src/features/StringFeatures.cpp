#include <string.h>
#include <assert.h>

#include "features/StringFeatures.h"
#include "lib/common.h"

CStringFeatures::CStringFeatures() : CFeatures(0l), num_vectors(0), features(NULL)
{
}

CStringFeatures::~CStringFeatures()
{
	delete[] features;
}

CStringFeatures::CStringFeatures(const CStringFeatures & orig): CFeatures(orig), 
num_vectors(orig.num_vectors)
{
	if (orig.features)
	{
		features=new T_STRING[orig.num_vectors];
		assert(features);

		for (int i=0; i<num_vectors; i++)
		{
			features[i].string=new CHAR[orig.features[i].length];
			assert(features[i].string!=NULL);
			features[i].length=orig.features[i].length;
			memcpy(features[i].string, orig.features[i].string, sizeof(CHAR)*orig.features[i].length); 
		}
	}
}

/// get feature vector for sample num
CHAR* CStringFeatures::get_feature_vector(long num, long &len)
{
	assert(features!=NULL);
	assert(num<num_vectors);

	len=features[num].length;
	return features[num].string;
}


bool CStringFeatures::load(FILE* src)
{
	return false;
}

bool CStringFeatures::save(FILE* dest)
{
	return false;
}
