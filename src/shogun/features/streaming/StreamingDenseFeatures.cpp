/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Soumyajit De, Viktor Gal,
 *          Vladislav Horbatiuk, Weijie Lin, Sergey Lisitsyn, Sanuj Sharma
 */

#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/features/streaming/StreamingDenseFeatures.h>
#include <shogun/io/streaming/StreamingFileFromDenseFeatures.h>

namespace shogun
{
template<class T>
StreamingDenseFeatures<T>::StreamingDenseFeatures() :
		StreamingDotFeatures()
{
	set_read_functions();
	init();
	parser.set_free_vector_after_release(false);
}

template<class T>
StreamingDenseFeatures<T>::StreamingDenseFeatures(std::shared_ptr<StreamingFile> file,
		bool is_labelled, int32_t size) :
		StreamingDotFeatures()
{
	init(file, is_labelled, size);
	set_read_functions();
	parser.set_free_vector_after_release(false);
}

template<class T> StreamingDenseFeatures<T>::StreamingDenseFeatures(
		std::shared_ptr<DenseFeatures<T>> dense_features, float64_t* lab) :
		StreamingDotFeatures()
{
	REQUIRE(dense_features, "%s::StreamingDenseFeatures(): Features needed!\n")

	bool is_labelled;
	int32_t size=1024;

	is_labelled=lab;
	auto file=std::make_shared<StreamingFileFromDenseFeatures<T>>(dense_features, lab);
	init(file, is_labelled, size);
	set_read_functions();
	parser.set_free_vector_after_release(false);
	parser.set_free_vectors_on_destruct(false);
	seekable=true;
}

template<class T> StreamingDenseFeatures<T>::~StreamingDenseFeatures()
{
	SG_DEBUG("entering %s::~StreamingDenseFeatures()\n", get_name())
	/* needed to prevent double free memory errors */
	current_vector.vector=NULL;
	current_vector.vlen=0;
	SG_DEBUG("leaving %s::~StreamingDenseFeatures()\n", get_name())
}

template<class T> void StreamingDenseFeatures<T>::reset_stream()
{
	if (seekable)
	{
		std::static_pointer_cast<StreamingFileFromDenseFeatures<T>>(working_file)->reset_stream();
		if (parser.is_running())
			parser.end_parser();
		parser.exit_parser();
		parser.init(working_file, has_labels, 1);
		parser.set_free_vector_after_release(false);
		parser.set_free_vectors_on_destruct(false);
		parser.start_parser();
	}
}

template<class T> float32_t StreamingDenseFeatures<T>::dense_dot(
		const float32_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len==current_vector.vlen)
	float32_t result=0;

	for (int32_t i=0; i<current_vector.vlen; i++)
		result+=current_vector[i]*vec2[i];

	return result;
}

template<class T> float64_t StreamingDenseFeatures<T>::dense_dot(
		const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len==current_vector.vlen)
	float64_t result=0;

	for (int32_t i=0; i<current_vector.vlen; i++)
		result+=current_vector[i]*vec2[i];

	return result;
}

template<class T> void StreamingDenseFeatures<T>::add_to_dense_vec(
		float32_t alpha, float32_t* vec2, int32_t vec2_len, bool abs_val)
{
	ASSERT(vec2_len==current_vector.vlen)

	if (abs_val)
	{
		for (int32_t i=0; i<current_vector.vlen; i++)
			vec2[i]+=alpha*Math::abs(current_vector[i]);
	}
	else
	{
		for (int32_t i=0; i<current_vector.vlen; i++)
			vec2[i]+=alpha*current_vector[i];
	}
}

template<class T> void StreamingDenseFeatures<T>::add_to_dense_vec(
		float64_t alpha, float64_t* vec2, int32_t vec2_len, bool abs_val)
{
	ASSERT(vec2_len==current_vector.vlen)

	if (abs_val)
	{
		for (int32_t i=0; i<current_vector.vlen; i++)
			vec2[i]+=alpha*Math::abs(current_vector[i]);
	}
	else
	{
		for (int32_t i=0; i<current_vector.vlen; i++)
			vec2[i]+=alpha*current_vector[i];
	}
}

template<class T> int32_t StreamingDenseFeatures<T>::get_nnz_features_for_vector()
{
	return current_vector.vlen;
}

template<class T> int32_t StreamingDenseFeatures<T>::get_num_vectors() const
{
	return 1;
}

template<class T>
void StreamingDenseFeatures<T>::set_vector_reader()
{
	parser.set_read_vector(&StreamingFile::get_vector);
}

template<class T>
void StreamingDenseFeatures<T>::set_vector_and_label_reader()
{
	parser.set_read_vector_and_label(&StreamingFile::get_vector_and_label);
}

#define GET_FEATURE_TYPE(f_type, sg_type)				\
template<> EFeatureType StreamingDenseFeatures<sg_type>::get_feature_type() const \
{									\
	return f_type;							\
}

GET_FEATURE_TYPE(F_BOOL, bool)
GET_FEATURE_TYPE(F_CHAR, char)
GET_FEATURE_TYPE(F_BYTE, uint8_t)
GET_FEATURE_TYPE(F_BYTE, int8_t)
GET_FEATURE_TYPE(F_SHORT, int16_t)
GET_FEATURE_TYPE(F_WORD, uint16_t)
GET_FEATURE_TYPE(F_INT, int32_t)
GET_FEATURE_TYPE(F_UINT, uint32_t)
GET_FEATURE_TYPE(F_LONG, int64_t)
GET_FEATURE_TYPE(F_ULONG, uint64_t)
GET_FEATURE_TYPE(F_SHORTREAL, float32_t)
GET_FEATURE_TYPE(F_DREAL, float64_t)
GET_FEATURE_TYPE(F_LONGREAL, floatmax_t)
#undef GET_FEATURE_TYPE

template<class T>
void StreamingDenseFeatures<T>::init()
{
	working_file=NULL;
	seekable=false;

	/* needed to prevent double free memory errors */
	current_vector.vector=NULL;
	current_vector.vlen=-1;

	set_generic<T>();
}

template<class T>
void StreamingDenseFeatures<T>::init(std::shared_ptr<StreamingFile> file, bool is_labelled,
		int32_t size)
{
	init();
	has_labels=is_labelled;
	working_file=file;

	parser.init(file, is_labelled, size);
	seekable=false;
}

template<class T>
void StreamingDenseFeatures<T>::start_parser()
{
	if (!parser.is_running())
		parser.start_parser();
}

template<class T>
void StreamingDenseFeatures<T>::end_parser()
{
	parser.end_parser();
}

template<class T>
bool StreamingDenseFeatures<T>::get_next_example()
{
	SG_DEBUG("entering\n");
	bool ret_value;
	ret_value=(bool)parser.get_next_example(current_vector.vector,
			current_vector.vlen, current_label);

	SG_DEBUG("leaving\n");
	return ret_value;
}

template<class T>
SGVector<T> StreamingDenseFeatures<T>::get_vector()
{
	return current_vector;
}

template<class T>
float64_t StreamingDenseFeatures<T>::get_label()
{
	ASSERT(has_labels)

	return current_label;
}

template<class T>
void StreamingDenseFeatures<T>::release_example()
{
	parser.finalize_example();
}

template<class T>
int32_t StreamingDenseFeatures<T>::get_dim_feature_space() const
{
	return current_vector.vlen;
}

template<class T>
float32_t StreamingDenseFeatures<T>::dot(std::shared_ptr<StreamingDotFeatures> df)
{
	ASSERT(df)
	ASSERT(df->get_feature_type() == get_feature_type())
	ASSERT(df->get_feature_class() == get_feature_class())
	auto sf=std::dynamic_pointer_cast<StreamingDenseFeatures<T>>(df);

	SGVector<T> other_vector=sf->get_vector();

	return linalg::dot(current_vector, other_vector);
}

template<class T>
float32_t StreamingDenseFeatures<T>::dot(SGVector<T> sgvec1)
{
	int32_t len1;
	len1=sgvec1.vlen;

	if (len1!=current_vector.vlen)
		SG_ERROR(
				"Lengths %d and %d not equal while computing dot product!\n", len1, current_vector.vlen);

	return linalg::dot(current_vector, sgvec1);
}

template<class T>
int32_t StreamingDenseFeatures<T>::get_num_features()
{
	return current_vector.vlen;
}

template<class T>
EFeatureClass StreamingDenseFeatures<T>::get_feature_class() const
{
	return C_STREAMING_DENSE;
}

template<class T>
std::shared_ptr<Features> StreamingDenseFeatures<T>::get_streamed_features(
		index_t num_elements)
{
	SG_DEBUG("entering\n");
	SG_DEBUG("Streaming %d elements\n", num_elements)

	REQUIRE(num_elements>0, "Requested number of feature vectors (%d) must be "
			"positive\n", num_elements);

	/* init matrix empty, as we dont know the dimension yet */
	SGMatrix<T> matrix;

	for (index_t i=0; i<num_elements; ++i)
	{
		/* check if we run out of data */
		if (!get_next_example())
		{
			SG_WARNING("Ran out of streaming data, reallocating matrix and "
					"returning!\n");

			/* allocating space for data so far, not this mighe be 0 bytes */
			SGMatrix<T> so_far(matrix.num_rows, i);

			/* copy */
			sg_memcpy(so_far.matrix, matrix.matrix,
					so_far.num_rows*so_far.num_cols*sizeof(T));

			matrix=so_far;
			break;
		}
		else
		{
			/* allocate matrix memory in first iteration */
			if (!matrix.matrix)
			{
				SG_DEBUG("Allocating %dx%d matrix\n",
						current_vector.vlen, num_elements);
				matrix=SGMatrix<T>(current_vector.vlen, num_elements);
			}

			/* get an example from stream and copy to feature matrix */
			SGVector<T> vec=get_vector();

			/* check for inconsistent dimensions */
			REQUIRE(vec.vlen==matrix.num_rows,
					"Dimension of streamed vector (%d) does not match "
					"dimensions of previous vectors (%d)\n",
					vec.vlen, matrix.num_rows);

			/* copy vector into matrix */
			sg_memcpy(&matrix.matrix[current_vector.vlen*i], vec.vector,
					vec.vlen*sizeof(T));

			/* clean up */
			release_example();
		}

	}

	SG_DEBUG("leaving returning %dx%d matrix\n", matrix.num_rows,
			matrix.num_cols);


	/* create new feature object from collected data */
	return std::make_shared<DenseFeatures<T>>(matrix);
}

template class StreamingDenseFeatures<bool> ;
template class StreamingDenseFeatures<char> ;
template class StreamingDenseFeatures<int8_t> ;
template class StreamingDenseFeatures<uint8_t> ;
template class StreamingDenseFeatures<int16_t> ;
template class StreamingDenseFeatures<uint16_t> ;
template class StreamingDenseFeatures<int32_t> ;
template class StreamingDenseFeatures<uint32_t> ;
template class StreamingDenseFeatures<int64_t> ;
template class StreamingDenseFeatures<uint64_t> ;
template class StreamingDenseFeatures<float32_t> ;
template class StreamingDenseFeatures<float64_t> ;
template class StreamingDenseFeatures<floatmax_t> ;
}
