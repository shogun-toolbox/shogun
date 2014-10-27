#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGNDArray.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(SGNDArrayTest,constructor)
{
	float64_t* data = SG_MALLOC(float64_t, 4);
	data[0] = 0;
	data[1] = 1;
	data[2] = 2;
	data[3] = 3;
	index_t *d_0 = SG_MALLOC(index_t, 2);
	d_0[0] = 2;
	d_0[1] = 2;
	SGNDArray<float64_t> arr_0(data, d_0, 2);
	EXPECT_EQ(arr_0.len_array, 4);
	EXPECT_EQ(arr_0.num_dims, 2);
	for (int i=0; i < 4; ++i)
		EXPECT_EQ(arr_0[i], i);

	index_t *d_1 = SG_MALLOC(index_t, 3);
	d_1[0] = 2;
	d_1[1] = 2;
	d_1[2] = 2;
	SGNDArray<float64_t> arr_1(d_1, 3);
	EXPECT_EQ(arr_1.len_array, 8);
	EXPECT_EQ(arr_1.num_dims, 3);

	arr_1.set_const(0);
	for (int i=0; i < 8; ++i)
		EXPECT_EQ(0, arr_1[i]);

	arr_1.set_const(3.3);
	for (int i=0; i < 8; ++i)
		EXPECT_EQ(3.3, arr_1[i]);

	/* test clone */
	SGNDArray<float64_t> arr_1_clone = arr_1.clone();
	for (int i=0; i < 8; ++i)
		EXPECT_EQ(arr_1_clone[i], arr_1[i]);

	/* test copy constructor */
	SGNDArray<float64_t> arr_2(arr_1_clone);
	EXPECT_EQ(arr_2.len_array, 8);
	for (int i=0; i < 8; ++i)
		EXPECT_EQ(arr_2[i], arr_1[i]);

	SGVector<index_t> d_3(3);
	d_3[0] = 2;
	d_3[1] = 2;
	d_3[2] = 2;
	SGNDArray<float64_t> arr_3(d_3);
	EXPECT_EQ(arr_3.len_array, 8);
	EXPECT_EQ(arr_3.num_dims, 3); 
}

TEST(SGNDArrayTest,setget)
{
	SGVector<index_t> v1(2);
	v1[0]=2;
	v1[1]=2;

	SGNDArray<float64_t> arr(v1);
	arr[0] = 0;
	arr[1] = 1;
	arr[2] = 2;
	arr[3] = 3;

	SGVector<index_t> v2 = arr.get_dimensions();
	EXPECT_EQ(v2.size(), arr.num_dims);
	for (int i=0; i < 2; i++)
		EXPECT_EQ(v2[i], v1[1]);

	for (int i=0; i < 4; i++)
		EXPECT_EQ(arr[i], i);

	/* get value */
	SGVector<index_t> index(2);
	index[0] = 0;
	index[1] = 0;
	float64_t val = arr.get_value(index);
	EXPECT_EQ(val, 0);
	index[0] = 0;
	index[1] = 1;
	val = arr.get_value(index);
	EXPECT_EQ(val, 1);
	index[0] = 1;
	index[1] = 0;
	val = arr.get_value(index);
	EXPECT_EQ(val, 2);
	index[0] = 1;
	index[1] = 1;
	val = arr.get_value(index);
	EXPECT_EQ(val, 3);
}

TEST(SGNDArrayTest, operators)
{
	SGVector<index_t> dims(3);
	dims[0] = 2;
	dims[1] = 2;
	dims[2] = 2;
	SGNDArray<float64_t> arr_0(dims);
	arr_0.set_const(1.0);

	SGNDArray<float64_t> arr_1 = arr_0.clone();
	arr_1.set_const(2.0);

	/* += */
	arr_1 += arr_0;
	for (int i=0; i < 8; i++)
		EXPECT_EQ(arr_1[i], 3.0);

	/* -= */
	arr_1 -= arr_0;
	for (int i=0; i < 8; i++)
		EXPECT_EQ(arr_1[i], 2.0);

	/* *= */
	arr_1 *= 2.0;
	for (int i=0; i < 8; i++)
		EXPECT_EQ(arr_1[i], 4.0);
}

TEST(SGNDArrayTest,max_element)
{
	SGVector<index_t> v1(2);
	v1[0]=2;
	v1[1]=2;

	SGNDArray<float64_t> arr(v1);
	arr[0] = 0;
	arr[1] = 1;
	arr[2] = 2;
	arr[3] = 3;

	index_t max_at;
	float64_t max_val;
	max_val = arr.max_element(max_at);
	EXPECT_EQ(max_val, 3);
	EXPECT_EQ(max_at, 3);

	arr[0] = 4;
	max_val = arr.max_element(max_at);
	EXPECT_EQ(max_val, 4);
	EXPECT_EQ(max_at, 0);

	arr[2] = 4;
	max_val = arr.max_element(max_at);
	EXPECT_EQ(max_val, 4);
	EXPECT_EQ(max_at, 2);
}

TEST(SGNDArrayTest,next_index)
{
	SGVector<index_t> dims(2);
	dims[0] = 2;
	dims[1] = 2;
	SGNDArray<float64_t> arr(dims);
	arr.set_const(0);

	SGVector<index_t> index(2);
	index[0] = 0;
	index[1] = 0;
	arr.next_index(index);
	EXPECT_EQ(index[0], 0);
	EXPECT_EQ(index[1], 1);
	arr.next_index(index);
	EXPECT_EQ(index[0], 1);
	EXPECT_EQ(index[1], 0);
	arr.next_index(index);
	EXPECT_EQ(index[0], 1);
	EXPECT_EQ(index[1], 1);

	SGVector<index_t> dims_3(3);
	dims_3[0] = 2;
	dims_3[1] = 2;
	dims_3[2] = 2;
	SGNDArray<float64_t> arr_3d(dims_3);
	arr_3d.set_const(0);

	SGVector<index_t> index_3(3);
	index_3[0] = 0;
	index_3[1] = 0;
	index_3[2] = 0;
	arr_3d.next_index(index_3);
	EXPECT_EQ(index_3[0], 0);
	EXPECT_EQ(index_3[1], 0);
	EXPECT_EQ(index_3[2], 1);
	arr_3d.next_index(index_3);
	EXPECT_EQ(index_3[0], 0);
	EXPECT_EQ(index_3[1], 1);
	EXPECT_EQ(index_3[2], 0);
	arr_3d.next_index(index_3);
	EXPECT_EQ(index_3[0], 0);
	EXPECT_EQ(index_3[1], 1);
	EXPECT_EQ(index_3[2], 1);
	arr_3d.next_index(index_3);
	EXPECT_EQ(index_3[0], 1);
	EXPECT_EQ(index_3[1], 0);
	EXPECT_EQ(index_3[2], 0);
}

TEST(SGNDArrayTest,expand)
{
	SGVector<index_t> dims_s(2);
	dims_s[0] = 2;
	dims_s[1] = 2;
	SGNDArray<float64_t> arr_s(dims_s);
	for (int i=0; i < 4; i++)
		arr_s[i] = i;
	
	SGVector<index_t> dims_l(3);
	dims_l[0] = 2;
	dims_l[1] = 2;
	dims_l[2] = 2;
	SGNDArray<float64_t> arr_l(dims_l);
	arr_l.set_const(-2);

	SGVector<index_t> axes(2);
	axes[0] = 0;
	axes[1] = 1;
	arr_s.expand(arr_l, axes);

	SGVector<index_t> index(3);
	index[0] = 0;
	index[1] = 0;
	index[2] = 0;

	float64_t val = arr_l.get_value(index);
	EXPECT_EQ(val, 0);
	
	index[0] = 0;
	index[1] = 0;
	index[2] = 1;
	val = arr_l.get_value(index);
	EXPECT_EQ(val, 0);

	index[0] = 0;
	index[1] = 1;
	index[2] = 0;
	val = arr_l.get_value(index);
	EXPECT_EQ(val, 1);

	index[0] = 0;
	index[1] = 1;
	index[2] = 1;
	val = arr_l.get_value(index);
	EXPECT_EQ(val, 1);

	index[0] = 1;
	index[1] = 0;
	index[2] = 0;
	val = arr_l.get_value(index);
	EXPECT_EQ(val, 2);

	index[0] = 1;
	index[1] = 0;
	index[2] = 1;
	val = arr_l.get_value(index);
	EXPECT_EQ(val, 2);

	index[0] = 1;
	index[1] = 1;
	index[2] = 0;
	val = arr_l.get_value(index);
	EXPECT_EQ(val, 3);

	index[0] = 1;
	index[1] = 1;
	index[2] = 1;
	val = arr_l.get_value(index);
	EXPECT_EQ(val, 3);
}
