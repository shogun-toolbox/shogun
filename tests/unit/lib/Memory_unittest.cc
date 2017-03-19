#include <shogun/lib/memory.h>
#include <shogun/mathematics/Math.h>
#include <gtest/gtest.h>

namespace shogun { template <class T> class SGMatrix; }
namespace shogun { template <class T> class SGSparseVector; }
namespace shogun { template <class T> class SGVector; }

using namespace shogun;

TEST(MemoryTest,get_copy)
{
	const int len = 3;
	unsigned char bytes[len] = {1,2,3};
	unsigned char* copy = (unsigned char*) get_copy(bytes, len);

	for (int i=0; i<len; i++)
		EXPECT_EQ(bytes[i], copy[i]);

	SG_FREE(copy);
}

TEST(MemoryTest,get_strdup)
{
	const char* str1 = "Test me crazy!";
	char* str2 = get_strdup(str1);

	int len1 = strlen(str1);
	int len2 = strlen(str2);
	EXPECT_EQ(len1, len2);

	for (int i=0; i<len1; i++)
		EXPECT_EQ(str1[i], str2[i]);

	SG_FREE(str2);
}

TEST(MemoryTest,SGVector)
{
	SGVector<float64_t>* v = SG_MALLOC(SGVector<float64_t>, 3);
	EXPECT_NE((SGVector<float64_t>*) NULL, v);
	SG_FREE(v);
}

TEST(MemoryTest,SGSparseVector)
{
	SGSparseVector<float64_t>* v = SG_MALLOC(SGSparseVector<float64_t>, 3);
	EXPECT_NE((SGSparseVector<float64_t>*) NULL, v);
	SG_FREE(v);
}

TEST(MemoryTest,SGMatrix)
{
	SGMatrix<float64_t>* m = SG_MALLOC(SGMatrix<float64_t>, 3);
	EXPECT_NE((SGMatrix<float64_t>*) NULL, m);
	SG_FREE(m);
}

template <typename T>
static void clone(T* dest, T const * const src, size_t size)
{
	shogun::sg_memcpy(dest, src, size);
}

TEST(MemoryTest, sg_memcpy)
{
	const index_t size = 10;
	auto src = SG_CALLOC(float64_t, size);
	for (index_t i=0; i<size; ++i)
		src[i]=CMath::randn_double();

	auto dest = SG_CALLOC(float64_t, size);

	clone(dest, src, size*sizeof(float64_t));

	for (index_t i=0; i<size; ++i)
		EXPECT_NEAR(src[i], dest[i], 1E-15);

	SG_FREE(src);
	SG_FREE(dest);
}
