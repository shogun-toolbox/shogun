/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Vojtech Franc, Soeren Sonnenburg
 * Written (W) 2013 Evangelos Anagnostopoulos
 * Copyright (C) 2010 Vojtech Franc, xfrancv@cmp.felk.cvut.cz
 * Copyright (C) 2010 Berlin Institute of Technology
 */
#include <shogun/features/LBPPyrDotFeatures.h>

using namespace shogun;

#define LIBLBP_INDEX(ROW,COL,NUM_ROWS) ((COL)*(NUM_ROWS)+(ROW))

CLBPPyrDotFeatures::CLBPPyrDotFeatures() : CDotFeatures()
{
	init(NULL, 0, 0);
	vec_nDim = 0;
}

CLBPPyrDotFeatures::CLBPPyrDotFeatures(CDenseFeatures<uint32_t>* image_set, int32_t image_w,
	int32_t image_h, uint16_t num_pyramids) : CDotFeatures()
{
	ASSERT(image_set)
	init(image_set, image_w, image_h);
	vec_nDim = liblbp_pyr_get_dim(num_pyramids);
}

void CLBPPyrDotFeatures::init(CDenseFeatures<uint32_t>* image_set, int32_t image_w, int32_t image_h)
{
	images = image_set;
	SG_REF(images);
	image_width = image_w;
	image_height = image_h;

	SG_ADD((CSGObject**) &images, "images", "Set of images", MS_NOT_AVAILABLE);
	SG_ADD(&image_width, "image_width", "The image width", MS_NOT_AVAILABLE);
	SG_ADD(&image_height, "image_height", "The image height", MS_NOT_AVAILABLE);
	SG_ADD(&vec_nDim, "vec_nDim", "The dimension of the pyr", MS_NOT_AVAILABLE);
}

CLBPPyrDotFeatures::~CLBPPyrDotFeatures()
{
	SG_UNREF(images);
}

CLBPPyrDotFeatures::CLBPPyrDotFeatures(const CLBPPyrDotFeatures& orig)
{
	init(orig.images, orig.image_width, orig.image_height);
	vec_nDim = orig.vec_nDim;
}

int32_t CLBPPyrDotFeatures::get_dim_feature_space() const
{
	return vec_nDim;
}

int32_t CLBPPyrDotFeatures::get_nnz_features_for_vector(int32_t num)
{
	return vec_nDim;
}

EFeatureType CLBPPyrDotFeatures::get_feature_type() const
{
	return F_UNKNOWN;
}

EFeatureClass CLBPPyrDotFeatures::get_feature_class() const
{
	return C_POLY;
}

int32_t CLBPPyrDotFeatures::get_num_vectors() const
{
	return images->get_num_vectors();
}

void* CLBPPyrDotFeatures::get_feature_iterator(int32_t vector_index)
{
	SG_NOTIMPLEMENTED
	return NULL;
}

bool CLBPPyrDotFeatures::get_next_feature(int32_t& index, float64_t& value, void* iterator)
{
	SG_NOTIMPLEMENTED
	return false;
}

void CLBPPyrDotFeatures::free_feature_iterator(void* iterator)
{
	SG_NOTIMPLEMENTED
}

float64_t CLBPPyrDotFeatures::dot(int32_t vec_idx1, CDotFeatures* df, int32_t vec_idx2)
{
	ASSERT(strcmp(df->get_name(),get_name())==0)
	CLBPPyrDotFeatures* lbp_feat = (CLBPPyrDotFeatures* ) df;
	ASSERT(get_dim_feature_space() == lbp_feat->get_dim_feature_space());

	SGVector<char> vec1 = get_transformed_image(vec_idx1);
	SGVector<char> vec2 = lbp_feat->get_transformed_image(vec_idx2);

	return SGVector<char>::dot(vec1.vector, vec2.vector, vec_nDim);
}

SGVector<char> CLBPPyrDotFeatures::get_transformed_image(int32_t index)
{
	SGVector<char> vec(vec_nDim);
	SGVector<char>::fill_vector(vec, vec_nDim, 0);

	int32_t ww;
	int32_t hh;
	uint32_t* img = get_image(index, ww, hh);

	int32_t offset = 0;
	while (true)
	{
		for (int32_t x=1; x<ww-1; x++)
		{
			for (int32_t y=1; y<hh-1; y++)
			{
				uint8_t pattern = create_lbp_pattern(img, x, y);
				vec[offset+pattern]++;
				offset += 256;
			}
		}
		if (vec_nDim <= offset)
			break;


		if (ww % 2 == 1)
			ww--;
		if (hh % 2 == 1)
			hh--;

		ww = ww/2;
		for (int32_t x=0; x<ww; x++)
			for (int32_t j=0; j<hh; j++)
				img[LIBLBP_INDEX(j,x,image_height)] = img[LIBLBP_INDEX(j,2*x,image_height)] +
					img[LIBLBP_INDEX(j,2*x+1,image_height)];

		hh = hh/2;
		for (int32_t y=0; y<hh; y++)
			for (int32_t j=0; j<ww; j++)
				img[LIBLBP_INDEX(y,j,image_height)] = img[LIBLBP_INDEX(2*y,j,image_height)] +
					img[LIBLBP_INDEX(2*y+1,j,image_height)];
	}

	SG_FREE(img);
	return vec;
}

uint32_t* CLBPPyrDotFeatures::get_image(int32_t index, int32_t& width, int32_t& height)
{
	int32_t len;
	bool do_free;
	uint32_t* image = images->get_feature_vector(index, len, do_free);
	uint32_t* img;
	img = SG_MALLOC(uint32_t, len);
	memcpy(img, image, len * sizeof(uint32_t));
	images->free_feature_vector(image, index, do_free);
	width = image_width;
	height = image_height;
	return img;
}

float64_t CLBPPyrDotFeatures::dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	if (vec2_len != vec_nDim)
		SG_ERROR("Dimensions don't match, vec2_dim=%d, vec_nDim=%d\n", vec2_len, vec_nDim)

	int32_t ww;
	int32_t hh;
	uint32_t* img = get_image(vec_idx1, ww, hh);

	float64_t dot_prod = 0;
	int32_t offset = 0;
	while (true)
	{
		for (int32_t x=1; x<ww-1; x++)
		{
			for (int32_t y=1; y<hh-1; y++)
			{
				uint8_t pattern = create_lbp_pattern(img, x, y);
				dot_prod += vec2[offset+pattern];
				offset += 256;
			}
		}
		if (vec_nDim <= offset)
			break;


		if (ww % 2 == 1)
			ww--;
		if (hh % 2 == 1)
			hh--;

		ww = ww/2;
		for (int32_t x=0; x<ww; x++)
			for (int32_t j=0; j<hh; j++)
				img[LIBLBP_INDEX(j,x,image_height)] = img[LIBLBP_INDEX(j,2*x,image_height)] +
					img[LIBLBP_INDEX(j,2*x+1,image_height)];

		hh = hh/2;
		for (int32_t y=0; y<hh; y++)
			for (int32_t j=0; j<ww; j++)
				img[LIBLBP_INDEX(y,j,image_height)] = img[LIBLBP_INDEX(2*y,j,image_height)] +
					img[LIBLBP_INDEX(2*y+1,j,image_height)];
	}

	SG_FREE(img);
	return dot_prod;
}

void CLBPPyrDotFeatures::add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val)
{
	if (vec2_len != vec_nDim)
		SG_ERROR("Dimensions don't match, vec2_dim=%d, vec_nDim=%d\n", vec2_len, vec_nDim)

	int32_t ww;
	int32_t hh;
	uint32_t* img = get_image(vec_idx1, ww, hh);

	if (abs_val)
		alpha = CMath::abs(alpha);

	int32_t offset = 0;

	while (true)
	{
		for (int32_t x=1; x<ww-1; x++)
		{
			for (int32_t y=1; y<hh-1; y++)
			{
				uint8_t pattern = create_lbp_pattern(img, x, y);
				vec2[offset+pattern] += alpha;
				offset += 256;
			}
		}
		if (vec_nDim <= offset)
			break;


		if (ww % 2 == 1)
			ww--;
		if (hh % 2 == 1)
			hh--;

		ww = ww/2;
		for (int32_t x=0; x<ww; x++)
			for (int32_t j=0; j<hh; j++)
				img[LIBLBP_INDEX(j,x,image_height)] = img[LIBLBP_INDEX(j,2*x,image_height)] +
					img[LIBLBP_INDEX(j,2*x+1,image_height)];

		hh = hh/2;
		for (int32_t y=0; y<hh; y++)
			for (int32_t j=0; j<ww; j++)
				img[LIBLBP_INDEX(y,j,image_height)] = img[LIBLBP_INDEX(2*y,j,image_height)] +
					img[LIBLBP_INDEX(2*y+1,j,image_height)];
	}
	SG_FREE(img);
}

uint8_t CLBPPyrDotFeatures::create_lbp_pattern(uint32_t* img, int32_t x, int32_t y)
{
	uint8_t pattern = 0;
	uint32_t center = img[LIBLBP_INDEX(y,x,image_height)];

	if (img[LIBLBP_INDEX(y-1,x-1,image_height)] < center)
		pattern |= 0x01;
	if (img[LIBLBP_INDEX(y-1,x,image_height)] < center)
		pattern |= 0x02;
	if (img[LIBLBP_INDEX(y-1,x+1,image_height)] < center)
		pattern |= 0x04;
	if (img[LIBLBP_INDEX(y,x-1,image_height)] < center)
		pattern |= 0x08;
	if (img[LIBLBP_INDEX(y,x+1,image_height)] < center)
		pattern |= 0x10;
	if (img[LIBLBP_INDEX(y+1,x-1,image_height)] < center)
		pattern |= 0x20;
	if (img[LIBLBP_INDEX(y+1,x,image_height)] < center)
		pattern |= 0x40;
	if (img[LIBLBP_INDEX(y+1,x+1,image_height)] < center)
		pattern |= 0x80;

	return pattern;
}

CFeatures* CLBPPyrDotFeatures::duplicate() const
{
	// return new CLBPPyrDotFeatures(*this);
	SG_NOTIMPLEMENTED
	return NULL;
}

uint32_t CLBPPyrDotFeatures::liblbp_pyr_get_dim(uint16_t nPyramids)
{
  uint32_t N = 0;
  uint32_t w = image_width;
  uint32_t h = image_height;

  for (uint32_t i=0; (i<nPyramids) && (CMath::min(w,h)>=3); i++)
  {
    N += (w-2)*(h-2);

    if (w % 2)
		w--;
    if (h % 2)
		h--;

    w = w/2;
    h = h/2;
  }
  return 256*N;
}
