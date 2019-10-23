/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Vojtech Franc, Soeren Sonnenburg, Evangelos Anagnostopoulos,
 *          Vladislav Horbatiuk, Evgeniy Andreev, Viktor Gal, Weijie Lin,
 *          Evan Shelhamer, Bjoern Esser, Sergey Lisitsyn, Sanuj Sharma
 */
#include <shogun/features/LBPPyrDotFeatures.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

#include <utility>

using namespace shogun;

#define LIBLBP_INDEX(ROW,COL,NUM_ROWS) ((COL)*(NUM_ROWS)+(ROW))

LBPPyrDotFeatures::LBPPyrDotFeatures() : DotFeatures()
{
	init(NULL, 0, 0);
	vec_nDim = 0;
}

LBPPyrDotFeatures::LBPPyrDotFeatures(const std::shared_ptr<DenseFeatures<uint32_t>>& image_set, int32_t image_w,
	int32_t image_h, uint16_t num_pyramids) : DotFeatures()
{
	ASSERT(image_set)
	init(image_set, image_w, image_h);
	vec_nDim = liblbp_pyr_get_dim(num_pyramids);
}

void LBPPyrDotFeatures::init(std::shared_ptr<DenseFeatures<uint32_t>> image_set, int32_t image_w, int32_t image_h)
{
	images = std::move(image_set);

	image_width = image_w;
	image_height = image_h;

	SG_ADD((std::shared_ptr<SGObject>*) &images, "images", "Set of images");
	SG_ADD(&image_width, "image_width", "The image width");
	SG_ADD(&image_height, "image_height", "The image height");
	SG_ADD(&vec_nDim, "vec_nDim", "The dimension of the pyr");
}

LBPPyrDotFeatures::~LBPPyrDotFeatures()
{

}

LBPPyrDotFeatures::LBPPyrDotFeatures(const LBPPyrDotFeatures& orig)
{
	init(orig.images, orig.image_width, orig.image_height);
	vec_nDim = orig.vec_nDim;
}

int32_t LBPPyrDotFeatures::get_dim_feature_space() const
{
	return vec_nDim;
}

int32_t LBPPyrDotFeatures::get_nnz_features_for_vector(int32_t num) const
{
	return vec_nDim;
}

EFeatureType LBPPyrDotFeatures::get_feature_type() const
{
	return F_UNKNOWN;
}

EFeatureClass LBPPyrDotFeatures::get_feature_class() const
{
	return C_POLY;
}

int32_t LBPPyrDotFeatures::get_num_vectors() const
{
	return images->get_num_vectors();
}

void* LBPPyrDotFeatures::get_feature_iterator(int32_t vector_index)
{
	not_implemented(SOURCE_LOCATION);
	return NULL;
}

bool LBPPyrDotFeatures::get_next_feature(int32_t& index, float64_t& value, void* iterator)
{
	not_implemented(SOURCE_LOCATION);
	return false;
}

void LBPPyrDotFeatures::free_feature_iterator(void* iterator)
{
	not_implemented(SOURCE_LOCATION);
}

float64_t LBPPyrDotFeatures::dot(int32_t vec_idx1, std::shared_ptr<DotFeatures> df, int32_t vec_idx2) const
{
	ASSERT(strcmp(df->get_name(),get_name())==0)
	auto lbp_feat = std::static_pointer_cast<LBPPyrDotFeatures>(df);
	ASSERT(get_dim_feature_space() == lbp_feat->get_dim_feature_space());

	SGVector<char> vec1 = get_transformed_image(vec_idx1);
	SGVector<char> vec2 = lbp_feat->get_transformed_image(vec_idx2);

	return linalg::dot(vec1, vec2);
}

SGVector<char> LBPPyrDotFeatures::get_transformed_image(int32_t index) const
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

uint32_t* LBPPyrDotFeatures::get_image(int32_t index, int32_t& width, int32_t& height) const
{
	int32_t len;
	bool do_free;
	uint32_t* image = images->get_feature_vector(index, len, do_free);
	uint32_t* img;
	img = SG_MALLOC(uint32_t, len);
	sg_memcpy(img, image, len * sizeof(uint32_t));
	images->free_feature_vector(image, index, do_free);
	width = image_width;
	height = image_height;
	return img;
}

float64_t
LBPPyrDotFeatures::dot(int32_t vec_idx1, const SGVector<float64_t>& vec2) const
{
	require(
	    vec2.size() == vec_nDim,
	    "Dimensions don't match, vec2_dim={}, vec_nDim={}", vec2.vlen,
	    vec_nDim);

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

void LBPPyrDotFeatures::add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val) const
{
	if (vec2_len != vec_nDim)
		error("Dimensions don't match, vec2_dim={}, vec_nDim={}", vec2_len, vec_nDim);

	int32_t ww;
	int32_t hh;
	uint32_t* img = get_image(vec_idx1, ww, hh);

	if (abs_val)
		alpha = Math::abs(alpha);

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

uint8_t LBPPyrDotFeatures::create_lbp_pattern(uint32_t* img, int32_t x, int32_t y) const
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

std::shared_ptr<Features> LBPPyrDotFeatures::duplicate() const
{
	return std::make_shared<LBPPyrDotFeatures>(*this);
}

uint32_t LBPPyrDotFeatures::liblbp_pyr_get_dim(uint16_t nPyramids)
{
  uint32_t N = 0;
  uint32_t w = image_width;
  uint32_t h = image_height;

  for (uint32_t i=0; (i<nPyramids) && (Math::min(w,h)>=3); i++)
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
