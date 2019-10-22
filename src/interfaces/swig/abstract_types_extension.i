/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

%include "Latent.i"

#if defined(SWIGPERL)
 //PTZ121108 example of classifier in examples/undocumented/libshogun/classifier_latent_svm.cpp
 //extention to make use of Data,LatentModel
 //TODO:PTZ121108 put it in another file like  classifier_latent_svm.i or %include  examples/undocumented/libshogun/classifier_latent_svm.cpp
 //or find a clever way to wrap CLatenModel, Data  instanciation, bless({}, shogun::LatentModel)
 // is not enough and would need a new wrapper, but yet new LatentModel() is not working,
 // (with error: "cannot allocate an object of abstract type") ?
%inline %{
  namespace shogun {
#define HOG_SIZE 1488
    struct BoundingBox : public Data
    {
    BoundingBox(int32_t x, int32_t y) : Data(), x_pos(x), y_pos(y) {};
      int32_t x_pos, y_pos;
      virtual const char* get_name() const { return "BoundingBox"; }
    };
    struct HOGFeatures : public Data
    {
    HOGFeatures(int32_t w, int32_t h) : Data(), width(w), height(h) {};
      int32_t width, height;
      float64_t ***hog;
      virtual const char* get_name() const { return "HOGFeatures"; }
    };
    class ObjectDetector: public LatentModel
    {
    public:
      ObjectDetector() {};
    ObjectDetector(const sdt::shared_ptr<LatentFeatures>& feat, const std::shared_ptr<LatentLabels>& labels)
      : LatentModel(feat, labels) {};
      virtual ~ObjectDetector() {};
      virtual int32_t get_dim() const { return HOG_SIZE; };
      virtual std::shared_ptr<DotFeatures> get_psi_feature_vectors()
      {
	int32_t num_examples = this->get_num_vectors();
	int32_t dim = this->get_dim();
	SGMatrix<float64_t> psi_m(dim, num_examples);
	for (int32_t i = 0; i < num_examples; ++i)
	  {
	    auto hf = m_features->get_sample(i)->as<HOGFeatures>();
	    auto bb = m_labels->get_latent_label(i)->as<BoundingBox>();
	    sg_memcpy(psi_m.matrix+i*dim, hf->hog[bb->x_pos][bb->y_pos], dim*sizeof(float64_t));
	  }
        auto psi_feats = std::make_shared<DenseFeatures<float64_t>>(psi_m);
        return psi_feats;
      };
      virtual std::shared_ptr<Data> infer_latent_variable(const SGVector<float64_t>& w, index_t idx)
      {
	int32_t pos_x = 0, pos_y = 0;
	float64_t max_score = -Math::INFTY;
        auto hf = m_features->get_sample(idx)->as<HOGFeatures>();
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
	SG_DEBUG("{} {} {}", pos_x, pos_y, max_score);
        return std::make_shared<BoundingBox>(pos_x, pos_y);
      };
    };
  }
%}
#endif

