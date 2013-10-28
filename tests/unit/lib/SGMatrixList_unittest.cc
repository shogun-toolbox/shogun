#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGMatrixList.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/common.h>
#include <shogun/lib/memory.h>
#include <gtest/gtest.h>
#include <stdio.h>

using namespace shogun;

TEST(SGMatrixListTest,default_ctor)
{
	printf("SGVector\n");
	SGVector<float64_t>* x = SG_MALLOC(SGVector<float64_t>, 3);
	SG_FREE(x);

	printf("SGSparseVector\n");
	SGSparseVector<float64_t>* y = SG_MALLOC(SGSparseVector<float64_t>, 3);
	SG_FREE(y);

	printf("SGMatrix\n");
	SGMatrix<float64_t>* z = SG_MALLOC(SGMatrix<float64_t>, 3);
	SG_FREE(z);

	SGMatrixList<float64_t> a;

	EXPECT_EQ(NULL, a.matrix_list);
	EXPECT_EQ(0, a.num_matrices);
}

TEST(SGMatrixListTest,fixed_nmats_ctor)
{
	SGMatrixList<float64_t> a(17);

	EXPECT_NE((SGMatrix<float64_t>*) NULL,a.matrix_list);
	EXPECT_EQ(17, a.num_matrices);

	for (int i=0; i < 17; i++)
	{
		SGMatrix<float64_t> m=a.get_matrix(i);
		EXPECT_EQ(0, m.num_rows);
		EXPECT_EQ(0, m.num_cols);
	}

}

TEST(SGMatrixListTest,list_ctor)
{
	SGMatrix<float64_t>* matrices = SG_MALLOC(SGMatrix<float64_t>, 17);

	for (int i=0; i < 17; i++)
	{
		matrices[i]=SGMatrix<float64_t>(10,5);
		matrices[i].zero();
	}

	SGMatrixList<float64_t> ml(matrices, 17);

	for (int i=0; i < 17; i++)
	{
		SGMatrix<float64_t> m=ml[i];

		EXPECT_EQ(10, m.num_rows);
		EXPECT_EQ(5, m.num_cols);
		EXPECT_EQ(0, m[0]);
	}
}

TEST(SGMatrixListTest,setget)
{
	SGMatrix<float64_t>* matrices = SG_MALLOC(SGMatrix<float64_t>, 17);

	for (int i=0; i < 17; i++)
	{
		matrices[i]=SGMatrix<float64_t>(10,5);
		matrices[i].zero();
	}

	SGMatrixList<float64_t> ml(matrices, 17);

	for (int i=0; i < 17; i++)
	{
		SGMatrix<float64_t> m=ml[i];

		EXPECT_EQ(10, m.num_rows);
		EXPECT_EQ(5, m.num_cols);
		EXPECT_EQ(0, m[0]);
	}

	for (int i=0; i < 7; i++)
	{
		SGMatrix<float64_t> m(3,2);
		m.set_const(17);
		ml.set_matrix(i, m);
		ml.set_matrix(i, m);
	}

	for (int i=0; i < 17; i++)
	{
		SGMatrix<float64_t> m=ml[i];

		if (i<7)
		{
			EXPECT_EQ(3, m.num_rows);
			EXPECT_EQ(2, m.num_cols);
			EXPECT_EQ(17, m[0]);
		}
		else
		{
			EXPECT_EQ(10, m.num_rows);
			EXPECT_EQ(5, m.num_cols);
			EXPECT_EQ(0, m[0]);
		}
	}
}

TEST(SGMatrixListTest,split)
{
	SGMatrix<int> v(2,3);
	v(0,0)=1;
	v(0,1)=2;
	v(0,2)=3;
	v(1,0)=3;
	v(1,1)=4;
	v(1,2)=6;

	SGMatrixList<int> ml=SGMatrixList<int>::split(v,3);

	EXPECT_EQ(3, ml.num_matrices);
	SGMatrix<int> m0=ml[0];
	SGMatrix<int> m1=ml[1];
	SGMatrix<int> m2=ml[2];
	//EXPECT_EQ(ml.num_matrices, 3);
}
