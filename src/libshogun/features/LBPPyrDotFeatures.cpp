/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Vojtech Franc, Soeren Sonnenburg
 * Copyright (C) 2010 Vojtech Franc, xfrancv@cmp.felk.cvut.cz
 * Copyright (C) 2010 Berlin Institute of Technology
 */
#include "features/LBPPyrDotFeatures.h"

using namespace shogun;

#define LIBLBP_INDEX(ROW,COL,NUM_ROWS) ((COL)*(NUM_ROWS)+(ROW))

//void CLBPPyrDotFeatures::liblbp_pyr_features(char *vec)
//{
//  uint32_t offset, ww, hh, x, y,center,j ;
//  uint8_t pattern;
//
//  offset=0;
///*  ww=win_W;*/
///*  hh=win_H;*/
//  ww=img_nCols;
//  hh=img_nRows;
//  while(1)
//  {
//    for(x=1; x < ww-1; x++)
//    {
//      for(y=1; y< hh-1; y++)
//      {
//        pattern = 0;
//        center = img[LIBLBP_INDEX(y,x,img_nRows)];
//        if(img[LIBLBP_INDEX(y-1,x-1,img_nRows)] < center) pattern = pattern | 0x01;
//        if(img[LIBLBP_INDEX(y-1,x,img_nRows)] < center)   pattern = pattern | 0x02;
//        if(img[LIBLBP_INDEX(y-1,x+1,img_nRows)] < center) pattern = pattern | 0x04;
//        if(img[LIBLBP_INDEX(y,x-1,img_nRows)] < center)   pattern = pattern | 0x08;
//        if(img[LIBLBP_INDEX(y,x+1,img_nRows)] < center)   pattern = pattern | 0x10;
//        if(img[LIBLBP_INDEX(y+1,x-1,img_nRows)] < center) pattern = pattern | 0x20;
//        if(img[LIBLBP_INDEX(y+1,x,img_nRows)] < center)   pattern = pattern | 0x40;
//        if(img[LIBLBP_INDEX(y+1,x+1,img_nRows)] < center) pattern = pattern | 0x80;
//
//        vec[offset+pattern]++;
//        offset += 256; 
//
//      }
//    }
//    if(vec_nDim <= offset) 
//      return;
//
//    if(ww % 2 == 1) ww--;
//    if(hh % 2 == 1) hh--;
//
//    ww = ww/2;
//    for(x=0; x < ww; x++)
//      for(j=0; j < hh; j++)
//        img[LIBLBP_INDEX(j,x,img_nRows)] = img[LIBLBP_INDEX(j,2*x,img_nRows)] + 
//          img[LIBLBP_INDEX(j,2*x+1,img_nRows)];
//
//    hh = hh/2;
//    for(y=0; y < hh; y++)
//      for(j=0; j < ww; j++)
//        img[LIBLBP_INDEX(y,j,img_nRows)] = img[LIBLBP_INDEX(2*y,j,img_nRows)] + 
//          img[LIBLBP_INDEX(2*y+1,j,img_nRows)];
//    
//  }
//
//  return;
//}

//void CLBPPyrDotFeatures::liblbp_pyr_subvec(int64_t *vec, uint32_t vec_nDim, uint32_t *img, uint16_t img_nRows, uint16_t img_nCols)
//{
//  uint32_t offset, ww, hh, x, y,center,j ;
//  uint8_t pattern;
//
//  offset=0;
///*  ww=win_W;*/
///*  hh=win_H;*/
//  ww=img_nCols;
//  hh=img_nRows;
//  while(1)
//  {
//    for(x=1; x < ww-1; x++)
//    {
//      for(y=1; y< hh-1; y++)
//      {
//        pattern = 0;
//        center = img[LIBLBP_INDEX(y,x,img_nRows)];
//        if(img[LIBLBP_INDEX(y-1,x-1,img_nRows)] < center) pattern = pattern | 0x01;
//        if(img[LIBLBP_INDEX(y-1,x,img_nRows)] < center)   pattern = pattern | 0x02;
//        if(img[LIBLBP_INDEX(y-1,x+1,img_nRows)] < center) pattern = pattern | 0x04;
//        if(img[LIBLBP_INDEX(y,x-1,img_nRows)] < center)   pattern = pattern | 0x08;
//        if(img[LIBLBP_INDEX(y,x+1,img_nRows)] < center)   pattern = pattern | 0x10;
//        if(img[LIBLBP_INDEX(y+1,x-1,img_nRows)] < center) pattern = pattern | 0x20;
//        if(img[LIBLBP_INDEX(y+1,x,img_nRows)] < center)   pattern = pattern | 0x40;
//        if(img[LIBLBP_INDEX(y+1,x+1,img_nRows)] < center) pattern = pattern | 0x80;
//
//        vec[offset+pattern]--;
//        offset += 256; 
//
//      }
//    }
//    if(vec_nDim <= offset) 
//      return;
//
//    if(ww % 2 == 1) ww--;
//    if(hh % 2 == 1) hh--;
//
//    ww = ww/2;
//    for(x=0; x < ww; x++)
//      for(j=0; j < hh; j++)
//        img[LIBLBP_INDEX(j,x,img_nRows)] = img[LIBLBP_INDEX(j,2*x,img_nRows)] + 
//          img[LIBLBP_INDEX(j,2*x+1,img_nRows)];
//
//    hh = hh/2;
//    for(y=0; y < hh; y++)
//      for(j=0; j < ww; j++)
//        img[LIBLBP_INDEX(y,j,img_nRows)] = img[LIBLBP_INDEX(2*y,j,img_nRows)] + 
//          img[LIBLBP_INDEX(2*y+1,j,img_nRows)];
//    
//  }
//
//  return;
//}

CLBPPyrDotFeatures::CLBPPyrDotFeatures(void)
	: CDotFeatures()
{
	SG_UNSTABLE("CLBPPyrDotFeatures::CLBPPyrDotFeatures(void)", "\n");

	m_feat = NULL;

	img = NULL;
	img_nRows = 0;
	img_nCols = 0;
	vec_nDim = 0;
}

CLBPPyrDotFeatures::CLBPPyrDotFeatures(CSimpleFeatures<uint32_t>* images, uint16_t num_pyramids)
	: CDotFeatures()
{
	ASSERT(images);

	m_feat = images;
	SG_REF(m_feat);
	img=m_feat->get_feature_matrix(img_nRows, img_nCols);
	vec_nDim=liblbp_pyr_get_dim(num_pyramids);
}

CLBPPyrDotFeatures::~CLBPPyrDotFeatures()
{
	SG_UNREF(m_feat);
}

float64_t CLBPPyrDotFeatures::dot(int32_t vec_idx1, CDotFeatures* df, int32_t vec_idx2)
{
	SG_NOTIMPLEMENTED;
	return 0;
}

float64_t CLBPPyrDotFeatures::dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	if (vec2_len != vec_nDim)
		SG_ERROR("Dimensions don't match, vec2_dim=%d, vec_nDim=%d\n", vec2_len, vec_nDim);

	//int32_t vlen;
	//bool do_free;
	//uint32_t* vec=m_feat->get_feature_vector(i, vlen, vfree);

	//double CLBPPyrDotFeatures::liblbp_pyr_dotprod(double *vec, uint32_t vec_nDim, uint32_t *img, uint16_t img_nRows, uint16_t img_nCols)
	//{
	double dot_prod = 0;
	int32_t offset=0;
	int32_t ww, hh, x, y, j;
	uint32_t center;
	uint8_t pattern;

	/*  ww=win_W;*/
	/*  hh=win_H;*/
	ww=img_nCols;
	hh=img_nRows;
	while(1)
	{
		for(x=1; x < ww-1; x++)
		{
			for(y=1; y< hh-1; y++)
			{
				pattern = 0;
				center = img[LIBLBP_INDEX(y,x,img_nRows)];
				if (img[LIBLBP_INDEX(y-1,x-1,img_nRows)] < center) pattern |= 0x01;
				if (img[LIBLBP_INDEX(y-1,x,img_nRows)] < center)   pattern |= 0x02;
				if (img[LIBLBP_INDEX(y-1,x+1,img_nRows)] < center) pattern |= 0x04;
				if (img[LIBLBP_INDEX(y,x-1,img_nRows)] < center)   pattern |= 0x08;
				if (img[LIBLBP_INDEX(y,x+1,img_nRows)] < center)   pattern |= 0x10;
				if (img[LIBLBP_INDEX(y+1,x-1,img_nRows)] < center) pattern |= 0x20;
				if (img[LIBLBP_INDEX(y+1,x,img_nRows)] < center)   pattern |= 0x40;
				if (img[LIBLBP_INDEX(y+1,x+1,img_nRows)] < center) pattern |= 0x80;

				dot_prod += vec2[offset+pattern];
				offset += 256; 


			}
		}
		if(vec_nDim <= offset) 
			return(dot_prod);


		if(ww % 2 == 1) ww--;
		if(hh % 2 == 1) hh--;

		ww = ww/2;
		for(x=0; x < ww; x++)
			for(j=0; j < hh; j++)
				img[LIBLBP_INDEX(j,x,img_nRows)] = img[LIBLBP_INDEX(j,2*x,img_nRows)] + 
					img[LIBLBP_INDEX(j,2*x+1,img_nRows)];

		hh = hh/2;
		for(y=0; y < hh; y++)
			for(j=0; j < ww; j++)
				img[LIBLBP_INDEX(y,j,img_nRows)] = img[LIBLBP_INDEX(2*y,j,img_nRows)] + 
					img[LIBLBP_INDEX(2*y+1,j,img_nRows)];    
	}

	//m_feat->free_feature_vector(vec, vlen, do_free);
	return dot_prod;
}

void CLBPPyrDotFeatures::add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val)
{
	if (vec2_len != vec_nDim)
		SG_ERROR("Dimensions don't match, vec2_dim=%d, vec_nDim=%d\n", vec2_len, vec_nDim);

	//int32_t vlen;
	//bool do_free;
	//uint32_t* vec=m_feat->get_feature_vector(i, vlen, vfree);


//void CLBPPyrDotFeatures::liblbp_pyr_addvec(int64_t *vec, uint32_t vec_nDim, uint32_t *img, uint16_t img_nRows, uint16_t img_nCols)
//{
  int32_t offset, ww, hh, x, y, j;
  uint32_t center;
  uint8_t pattern;

  offset=0;
/*  ww=win_W;*/
/*  hh=win_H;*/
  ww=img_nCols;
  hh=img_nRows;
  while(1)
  {
    for(x=1; x < ww-1; x++)
    {
      for(y=1; y< hh-1; y++)
      {
        pattern = 0;
        center = img[LIBLBP_INDEX(y,x,img_nRows)];
        if(img[LIBLBP_INDEX(y-1,x-1,img_nRows)] < center) pattern |= 0x01;
        if(img[LIBLBP_INDEX(y-1,x,img_nRows)] < center)   pattern |= 0x02;
        if(img[LIBLBP_INDEX(y-1,x+1,img_nRows)] < center) pattern |= 0x04;
        if(img[LIBLBP_INDEX(y,x-1,img_nRows)] < center)   pattern |= 0x08;
        if(img[LIBLBP_INDEX(y,x+1,img_nRows)] < center)   pattern |= 0x10;
        if(img[LIBLBP_INDEX(y+1,x-1,img_nRows)] < center) pattern |= 0x20;
        if(img[LIBLBP_INDEX(y+1,x,img_nRows)] < center)   pattern |= 0x40;
        if(img[LIBLBP_INDEX(y+1,x+1,img_nRows)] < center) pattern |= 0x80;

        vec2[offset+pattern]+=alpha;
        offset += 256; 

      }
    }
    if(vec_nDim <= offset) 
      return;

    if(ww % 2 == 1) ww--;
    if(hh % 2 == 1) hh--;

    ww = ww/2;
    for(x=0; x < ww; x++)
      for(j=0; j < hh; j++)
        img[LIBLBP_INDEX(j,x,img_nRows)] = img[LIBLBP_INDEX(j,2*x,img_nRows)] + 
             img[LIBLBP_INDEX(j,2*x+1,img_nRows)];

    hh = hh/2;
    for(y=0; y < hh; y++)
      for(j=0; j < ww; j++)
        img[LIBLBP_INDEX(y,j,img_nRows)] = img[LIBLBP_INDEX(2*y,j,img_nRows)] + 
          img[LIBLBP_INDEX(2*y+1,j,img_nRows)];
    
  }

 // return;
//}

	//m_feat->free_feature_vector(vec, vlen, do_free);
}

CFeatures* CLBPPyrDotFeatures::duplicate() const
{
	return new CLBPPyrDotFeatures(*this);
}

uint32_t CLBPPyrDotFeatures::liblbp_pyr_get_dim(uint16_t nPyramids)
{
  uint32_t w, h, N, i;

  for(w=img_nCols, h=img_nRows, N=0, i=0; i < nPyramids && CMath::min(w,h) >= 3; i++)
  {
    N += (w-2)*(h-2);

    if(w % 2) w--;
    if(h % 2) h--;
    w = w/2;
    h = h/2;
  }
  return(256*N);
}
