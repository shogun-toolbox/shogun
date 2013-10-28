/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Written (W) 2012 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/mathematics/Math.h>
#include <shogun/features/streaming/StreamingDenseFeatures.h>
#include <shogun/io/streaming/StreamingFileFromDenseFeatures.h>

namespace shogun
{
template<class T>
CStreamingDenseFeatures<T>::CStreamingDenseFeatures() :
		CStreamingDotFeatures()
{
	set_read_functions();
	init();
	parser.set_free_vector_after_release(false);
}

template<class T>
CStreamingDenseFeatures<T>::CStreamingDenseFeatures(CStreamingFile* file,
		bool is_labelled, int32_t size) :
		CStreamingDotFeatures()
{
	init(file, is_labelled, size);
	set_read_functions();
	parser.set_free_vector_after_release(false);
}

template<class T> CStreamingDenseFeatures<T>::CStreamingDenseFeatures(
		CDenseFeatures<T>* dense_features, float64_t* lab) :
		CStreamingDotFeatures()
{
	REQUIRE(dense_features, "%s::CStreamingDenseFeatures(): Features needed!\n")

	CStreamingFileFromDenseFeatures<T>* file;
	bool is_labelled;
	int32_t size=1024;

	is_labelled=lab;
	file=new CStreamingFileFromDenseFeatures<T>(dense_features, lab);
	init(file, is_labelled, size);
	set_read_functions();
	parser.set_free_vector_after_release(false);
	parser.set_free_vectors_on_destruct(false);
	seekable=true;
}

template<class T> CStreamingDenseFeatures<T>::~CStreamingDenseFeatures()
{
	SG_DEBUG("entering %s::~CStreamingDenseFeatures()\n", get_name())
	/* needed to prevent double free memory errors */
	current_vector.vector=NULL;
	current_vector.vlen=0;
	SG_DEBUG("leaving %s::~CStreamingDenseFeatures()\n", get_name())
}

template<class T> void CStreamingDenseFeatures<T>::reset_stream()
{
	if (seekable)
	{
		((CStreamingFileFromDenseFeatures<T>*)working_file)->reset_stream();
		parser.exit_parser();
		parser.init(working_file, has_labels, 1);
		parser.set_free_vector_after_release(false);
		parser.start_parser();
	}
}

template<class T> float32_t CStreamingDenseFeatures<T>::dense_dot(
		const float32_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len==current_vector.vlen)
	float32_t result=0;

	for (int32_t i=0; i<current_vector.vlen; i++)
		result+=current_vector[i]*vec2[i];

	return result;
}

template<class T> float64_t CStreamingDenseFeatures<T>::dense_dot(
		const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len==current_vector.vlen)
	float64_t result=0;

	for (int32_t i=0; i<current_vector.vlen; i++)
		result+=current_vector[i]*vec2[i];

	return result;
}

template<class T> void CStreamingDenseFeatures<T>::add_to_dense_vec(
		float32_t alpha, float32_t* vec2, int32_t vec2_len, bool abs_val)
{
	ASSERT(vec2_len==current_vector.vlen)

	if (abs_val)
	{
		for (int32_t i=0; i<current_vector.vlen; i++)
			vec2[i]+=alpha*CMath::abs(current_vector[i]);
	}
	else
	{
		for (int32_t i=0; i<current_vector.vlen; i++)
			vec2[i]+=alpha*current_vector[i];
	}
}

template<class T> void CStreamingDenseFeatures<T>::add_to_dense_vec(
		float64_t alpha, float64_t* vec2, int32_t vec2_len, bool abs_val)
{
	ASSERT(vec2_len==current_vector.vlen)

	if (abs_val)
	{
		for (int32_t i=0; i<current_vector.vlen; i++)
			vec2[i]+=alpha*CMath::abs(current_vector[i]);
	}
	else
	{
		for (int32_t i=0; i<current_vector.vlen; i++)
			vec2[i]+=alpha*current_vector[i];
	}
}

template<class T> int32_t CStreamingDenseFeatures<T>::get_nnz_features_for_vector()
{
	return current_vector.vlen;
}

template<class T> CFeatures* CStreamingDenseFeatures<T>::duplicate() const
{
	return new CStreamingDenseFeatures<T>(*this);
}

template<class T> int32_t CStreamingDenseFeatures<T>::get_num_vectors() const
{
	return 1;
}

template<class T>
void CStreamingDenseFeatures<T>::set_vector_reader()
{
	parser.set_read_vector(&CStreamingFile::get_vector);
}

template<class T>
void CStreamingDenseFeatures<T>::set_vector_and_label_reader()
{
	parser.set_read_vector_and_label(&CStreamingFile::get_vector_and_label);
}

#define GET_FEATURE_TYPE(f_type, sg_type)				\
template<> EFeatureType CStreamingDenseFeatures<sg_type>::get_feature_type() const \
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
void CStreamingDenseFeatures<T>::init()
{
	working_file=NULL;
	seekable=false;

	/* needed to prevent double free memory errors */
	current_vector.vector=NULL;
	current_vector.vlen=-1;

	set_generic<T>();
}

template<class T>
void CStreamingDenseFeatures<T>::init(CStreamingFile* file, bool is_labelled,
		int32_t size)
{
	init();
	has_labels=is_labelled;
	working_file=file;
	SG_REF(working_file);
	parser.init(file, is_labelled, size);
	seekable=false;
}

template<class T>
void CStreamingDenseFeatures<T>::start_parser()
{
	if (!parser.is_running())
		parser.start_parser();
}

template<class T>
void CStreamingDenseFeatures<T>::end_parser()
{
	parser.end_parser();
}

template<class T>
bool CStreamingDenseFeatures<T>::get_next_example()
{
	bool ret_value;
	ret_value=(bool)parser.get_next_example(current_vector.vector,
			current_vector.vlen, current_label);

	return ret_value;
}

template<class T>
SGVector<T> CStreamingDenseFeatures<T>::get_vector()
{
	return current_vector;
}

template<class T>
float64_t CStreamingDenseFeatures<T>::get_label()
{
	ASSERT(has_labels)

	return current_label;
}

template<class T>
void CStreamingDenseFeatures<T>::release_example()
{
	parser.finalize_example();
}

template<class T>
int32_t CStreamingDenseFeatures<T>::get_dim_feature_space() const
{
	return current_vector.vlen;
}

template<class T>
float32_t CStreamingDenseFeatures<T>::dot(CStreamingDotFeatures* df)
{
	ASSERT(df)
	ASSERT(df->get_feature_type() == get_feature_type())
	ASSERT(df->get_feature_class() == get_feature_class())
	CStreamingDenseFeatures<T>* sf=(CStreamingDenseFeatures<T>*)df;

	SGVector<T> other_vector=sf->get_vector();

	return SGVector<T>::dot(current_vector.vector, other_vector.vector, current_vector.vlen);
}

template<class T>
float32_t CStreamingDenseFeatures<T>::dot(SGVector<T> sgvec1)
{
	int32_t len1;
	len1=sgvec1.vlen;

	if (len1!=current_vector.vlen)
		SG_ERROR(
				"Lengths %d and %d not equal while computing dot product!\n", len1, current_vector.vlen);

	return SGVector<T>::dot(current_vector.vector, sgvec1.vector, len1);
}

template<class T>
int32_t CStreamingDenseFeatures<T>::get_num_features()
{
	return current_vector.vlen;
}

template<class T>
EFeatureClass CStreamingDenseFeatures<T>::get_feature_class() const
{
	return C_STREAMING_DENSE;
}

template<class T>
CFeatures* CStreamingDenseFeatures<T>::get_streamed_features(
		index_t num_elements)
{
	SG_DEBUG("entering %s(%p)::get_streamed_features(%d)\n", get_name(), this,
			num_elements);

	/* init matrix empty since num_rows is not yet known */
	SGMatrix<T> matrix;

	for (index_t i=0; i<num_elements; ++i)
	{
		/* check if we run out of data */
		if (!get_next_example())
		{
			SG_WARNING("%s::get_streamed_features(): ran out of streaming "
					"data, reallocating matrix and returning!\n", get_name());

			/* allocating space for data so far */
			SGMatrix<T> so_far(matrix.num_rows, i);

			/* copy */
			memcpy(so_far.matrix, matrix.matrix,
					so_far.num_rows*so_far.num_cols*sizeof(T));

			matrix=so_far;
			break;
		}
		else
		{
			/* allocate matrix memory during first run */
			if (!matrix.matrix)
			{
				SG_DEBUG("%s::get_streamed_features(): allocating %dx%d matrix\n",
						get_name(), current_vector.vlen, num_elements);
				matrix=SGMatrix<T>(current_vector.vlen, num_elements);
			}

			/* get an example from stream and copy to feature matrix */
			SGVector<T> vec=get_vector();

			/* check for inconsistent dimensions */
			if (vec.vlen!=matrix.num_rows)
			{
				SG_ERROR("%s::get_streamed_features(): streamed vectors have "
						"different dimensions. This is not allowed!\n",
						get_name());
			}

			/* copy vector into matrix */
			memcpy(&matrix.matrix[current_vector.vlen*i], vec.vector,
					vec.vlen*sizeof(T));

			/* evtl output vector */
			if (sg_io->get_loglevel()==MSG_DEBUG)
			{
				SG_DEBUG("%d. ", i)
				vec.display_vector("streamed vector");
			}

			/* clean up */
			release_example();
		}

	}

	/* create new feature object from collected data */
	CDenseFeatures<T>* result=new CDenseFeatures<T>(matrix);

	SG_DEBUG("leaving %s(%p)::get_streamed_features(%d) and returning %dx%d "
			"matrix\n", get_name(), this, num_elements, matrix.num_rows,
			matrix.num_cols);

	return result;
}

template class CStreamingDenseFeatures<bool> ;
template class CStreamingDenseFeatures<char> ;
template class CStreamingDenseFeatures<int8_t> ;
template class CStreamingDenseFeatures<uint8_t> ;
template class CStreamingDenseFeatures<int16_t> ;
template class CStreamingDenseFeatures<uint16_t> ;
template class CStreamingDenseFeatures<int32_t> ;
template class CStreamingDenseFeatures<uint32_t> ;
template class CStreamingDenseFeatures<int64_t> ;
template class CStreamingDenseFeatures<uint64_t> ;
template class CStreamingDenseFeatures<float32_t> ;
template class CStreamingDenseFeatures<float64_t> ;
template class CStreamingDenseFeatures<floatmax_t> ;
}
