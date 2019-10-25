#include <shogun/base/ShogunEnv.h>
#include <shogun/base/progress.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/LatentFeatures.h>
#include <shogun/io/SGIO.h>
#include <shogun/labels/LatentLabels.h>
#include <shogun/latent/LatentSVM.h>
#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>

#include <libgen.h>

using namespace shogun;

#define MAX_LINE_LENGTH 4096
#define HOG_SIZE 1488

struct CBoundingBox : public Data
{
	CBoundingBox(int32_t x, int32_t y) : Data(), x_pos(x), y_pos(y) {};

	int32_t x_pos, y_pos;

	/** @return name of SGSerializable */
	virtual const char* get_name() const { return "BoundingBox"; }
};

struct CHOGFeatures : public Data
{
	CHOGFeatures(int32_t w, int32_t h) : Data(), width(w), height(h) {};

	int32_t width, height;
	float64_t ***hog;

	/** @return name of SGSerializable */
	virtual const char* get_name() const { return "HOGFeatures"; }
};

class CObjectDetector: public LatentModel
{
	public:
		CObjectDetector() {}
		CObjectDetector(LatentFeatures* feat, CLatentLabels* labels) : LatentModel(feat, labels) {}

		virtual ~CObjectDetector() {}

		virtual int32_t get_dim() const { return HOG_SIZE; }

		virtual DotFeatures* get_psi_feature_vectors()
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

			DenseFeatures<float64_t>* psi_feats = new DenseFeatures<float64_t>(psi_m);
			return psi_feats;
		}

		virtual Data* infer_latent_variable(const SGVector<float64_t>& w, index_t idx)
		{
			int32_t pos_x = 0, pos_y = 0;
			float64_t max_score = -Math::INFTY;

			CHOGFeatures* hf = (CHOGFeatures*) m_features->get_sample(idx);
			for (int i = 0; i < hf->width; ++i)
			{
				for (int j = 0; j < hf->height; ++j)
				{
					float64_t score = Math::dot(w.vector, hf->hog[i][j], w.vlen);

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

			return h;
		}

};

static void read_dataset(char* fname, LatentFeatures*& feats, CLatentLabels*& labels)
{
	FILE* fd = fopen(fname, "r");
	char line[MAX_LINE_LENGTH];
	char *pchar, *last_pchar;
	int num_examples,label,height,width;

	char* path = dirname(fname);

	if (fd == NULL)
		SG_SERROR("Cannot open input file %s!\n", fname);

	fgets(line, MAX_LINE_LENGTH, fd);
	num_examples = atoi(line);

	labels = new CLatentLabels(num_examples);

	BinaryLabels* ys = new BinaryLabels(num_examples);

	feats = new LatentFeatures(num_examples);

	auto pb = progress(range(num_examples));
	Math::init_random();
	for (int i = 0; (!feof(fd)) && (i < num_examples); ++i)
	{
		fgets(line, MAX_LINE_LENGTH, fd);

		pchar = line;
		while ((*pchar)!=' ') pchar++;
		*pchar = '\0';
		pchar++;

		/* label: {-1, 1} */
		last_pchar = pchar;
		while ((*pchar)!=' ') pchar++;
		*pchar = '\0';
		label = (atoi(last_pchar) % 2 == 0) ? 1 : -1;
		pchar++;

		if (ys->set_label(i, label) == false)
			SG_SERROR("Couldn't set label for element %d\n", i);

		last_pchar = pchar;
		while ((*pchar)!=' ') pchar++;
		*pchar = '\0';
		width = atoi(last_pchar);
		pchar++;

		last_pchar = pchar;
		while ((*pchar)!='\n') pchar++;
		*pchar = '\0';
		height = atoi(last_pchar);

		/* create latent label */
		int x = Math::random(0, width-1);
		int y = Math::random(0, height-1);
		CBoundingBox* bb = new CBoundingBox(x,y);
		labels->add_latent_label(bb);

		pb.print_progress();
		CHOGFeatures* hog = new CHOGFeatures(width, height);
		hog->hog = SG_CALLOC(float64_t**, hog->width);
		for (int j = 0; j < width; ++j)
		{
			hog->hog[j] = SG_CALLOC(float64_t*, hog->height);
			for (int k = 0; k < height; ++k)
			{
				char filename[MAX_LINE_LENGTH];
				hog->hog[j][k] = SG_CALLOC(float64_t, HOG_SIZE);

				sprintf(filename,"%s/%s.%03d.%03d.txt",path,line,j,k);
				FILE* f = fopen(filename, "r");
				if (f == NULL)
					SG_SERROR("Could not open file: %s\n", filename);
				for (int l = 0; l < HOG_SIZE; ++l)
					fscanf(f,"%lf",&hog->hog[j][k][l]);
				fclose(f);
			}
		}
		feats->add_sample(hog);
	}
	fclose(fd);

	labels->set_labels(ys);

	pb.complete();
}

int main(int argc, char** argv)
{
	env()->io()->set_loglevel(MSG_DEBUG);

	/* check whether the train/test args are given */
	if (argc < 3)
	{
		SG_SERROR("not enough arguements given\n");
	}

	LatentFeatures* train_feats = NULL;
	CLatentLabels* train_labels = NULL;
	/* read train data set */
	read_dataset(argv[1], train_feats, train_labels);

	/* train the classifier */
	float64_t C = 10.0;

	CObjectDetector* od = new CObjectDetector(train_feats, train_labels);
	LatentSVM llm(od, C);
	llm.train();

	//  LatentFeatures* test_feats = NULL;
	//  CLatentLabels* test_labels = NULL;
	//  read_dataset(argv[2], test_feats, test_labels);

	SG_SPRINT("Testing with the test set\n");
	llm.apply(train_feats);

	return 0;
}

