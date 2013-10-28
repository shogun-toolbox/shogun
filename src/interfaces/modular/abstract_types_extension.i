/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * this code is inspired from
 * examples/documented/libshogun/classifier_latent_svm.cpp
 * it serves the purpose of instanciating and wrapping shogun::CLatenModel.
 *
 * Written (W) 2012 Christian Montanari
 */

%include "Latent.i"
%rename(ObjectDetector) CObjectDetector;

#if defined(SWIGPERL)
 //PTZ121108 example of classifier in examples/undocumented/libshogun/classifier_latent_svm.cpp
 //extention to make use of CData,CLatentModel
 //TODO:PTZ121108 put it in another file like  classifier_latent_svm.i or %include  examples/undocumented/libshogun/classifier_latent_svm.cpp
 //or find a clever way to wrap CLatenModel, CData  instanciation, bless({}, modshogun::LatentModel)
 // is not enough and would need a new wrapper, but yet new CLatentModel() is not working,
 // (with error: "cannot allocate an object of abstract type") ?
%inline %{
  namespace shogun {
#define HOG_SIZE 1488
    struct CBoundingBox : public CData
    {
    CBoundingBox(int32_t x, int32_t y) : CData(), x_pos(x), y_pos(y) {};
      int32_t x_pos, y_pos;
      virtual const char* get_name() const { return "BoundingBox"; }
    };
    struct CHOGFeatures : public CData
    {
    CHOGFeatures(int32_t w, int32_t h) : CData(), width(w), height(h) {};
      int32_t width, height;
      float64_t ***hog;
      virtual const char* get_name() const { return "HOGFeatures"; }
    };
    class CObjectDetector: public CLatentModel
    {
    public:
      CObjectDetector() {};
    CObjectDetector(CLatentFeatures* feat, CLatentLabels* labels)
      : CLatentModel(feat, labels) {};
      virtual ~CObjectDetector() {};
      virtual int32_t get_dim() const { return HOG_SIZE; };
      virtual CDotFeatures* get_psi_feature_vectors()
      {
	int32_t num_examples = this->get_num_vectors();
	int32_t dim = this->get_dim();
	SGMatrix<float64_t> psi_m(dim, num_examples);
	for (int32_t i = 0; i < num_examples; ++i)
	  {
	    CHOGFeatures* hf = (CHOGFeatures*) m_features->get_sample(i);
	    CBoundingBox* bb = (CBoundingBox*) m_labels->get_latent_label(i);
	    memcpy(psi_m.matrix+i*dim, hf->hog[bb->x_pos][bb->y_pos], dim*sizeof(float64_t));
	  }
	CDenseFeatures<float64_t>* psi_feats = new CDenseFeatures<float64_t>(psi_m);
	return psi_feats;
      };
      virtual CData* infer_latent_variable(const SGVector<float64_t>& w, index_t idx)
      {
	int32_t pos_x = 0, pos_y = 0;
	float64_t max_score = -CMath::INFTY;
	CHOGFeatures* hf = (CHOGFeatures*) m_features->get_sample(idx);
	for (int i = 0; i < hf->width; ++i)
	  {
	    for (int j = 0; j < hf->height; ++j)
	      {
		float64_t score = w.dot(w.vector, hf->hog[i][j], w.vlen);
		if (score > max_score)
		  {
		    pos_x = i;
		    pos_y = j;
		    max_score = score;
		  }
	      }
	  }
	SG_SDEBUG("%d %d %f\n", pos_x, pos_y, max_score);
	CBoundingBox* h = new CBoundingBox(pos_x, pos_y);
	SG_REF(h);
	return h;
      };
    };
  }
%}
#endif

