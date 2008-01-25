#ifndef _LIBLINEAR_H
#define _LIBLINEAR_H

#include "lib/config.h"

#ifdef HAVE_LAPACK
#include "classifier/svm/Tron.h"
#include "features/SparseFeatures.h"

#ifdef __cplusplus
extern "C" {
#endif

/** problem */
struct problem
{
	/** l */
	INT l;
	/** n */
	INT n;
	/** y */
	INT *y;
	/** sparse features x */
	CSparseFeatures<DREAL>* x;
	/** if bias shall be used */
	bool use_bias;
};

/** parameter */
struct parameter
{
	/** solver type */
	int solver_type;

	/* these are for training only */
	/** stopping criteria */
	double eps;
	/** C */
	double C;
	/** number of weights */
	int nr_weight;
	/** weight label */
	int *weight_label;
	/** weight */
	double* weight;
};

/** model */
struct model
{
	/** parameter */
	struct parameter param;
	/** number of classes */
	int nr_class;
	/** number of features */
	int nr_feature;
	/** w */
	double *w;
	/** label of each class (label[n]) */
	int *label;
	/** bias */
	double bias;
};

struct model* train(const struct problem *prob, const struct parameter *param);
void cross_validation(const struct problem *prob, const struct parameter *param, int nr_fold, int *target);

int predict_values(const struct model *model_, const struct feature_node *x, double* dec_values);
int predict(const struct model *model_, const struct feature_node *x);
int predict_probability(const struct model *model_, const struct feature_node *x, double* prob_estimates);

int save_model(const char *model_file_name, const struct model *model_);
struct model *load_model(const char *model_file_name);

int get_nr_feature(const struct model *model_);
int get_nr_class(const struct model *model_);
void get_labels(const struct model *model_, int* label);

void destroy_model(struct model *model_);
void destroy_param(struct parameter *param);
const char *check_parameter(const struct problem *prob, const struct parameter *param);

#ifdef __cplusplus
}
#endif

/** class l2loss_svm_vun */
class l2loss_svm_fun : public function
{
public:
	/** constructor
	 *
	 * @param prob prob
	 * @param Cp Cp
	 * @param Cn Cn
	 */
	l2loss_svm_fun(const problem *prob, double Cp, double Cn);
	~l2loss_svm_fun();
	
	/** fun
	 *
	 * @param w w
	 * @return something floaty
	 */
	double fun(double *w);
	
	/** grad
	 *
	 * @param w w
	 * @param g g
	 */
	void grad(double *w, double *g);

	/** Hv
	 *
	 * @param s s
	 * @param Hs Hs
	 */
	void Hv(double *s, double *Hs);

	/** get number of variables
	 *
	 * @return number of variables
	 */
	int get_nr_variable(void);

private:
	void Xv(double *v, double *Xv);
	void subXv(double *v, double *Xv);
	void subXTv(double *v, double *XTv);

	double *C;
	double *z;
	double *D;
	int *I;
	int sizeI;
	const problem *prob;
};

/** class l2_lr_fun */
class l2_lr_fun : public function
{
public:
	/** constructor
	 *
	 * @param prob prob
	 * @param Cp Cp
	 * @param Cn Cn
	 */
	l2_lr_fun(const problem *prob, double Cp, double Cn);
	~l2_lr_fun();

	/** fun
	 *
	 * @param w w
	 * @return something floaty
	 */
	double fun(double *w);
	
	/** grad
	 *
	 * @param w w
	 * @param g g
	 */
	void grad(double *w, double *g);

	/** Hv
	 *
	 * @param s s
	 * @param Hs Hs
	 */
	void Hv(double *s, double *Hs);

	int get_nr_variable(void);

private:
	void Xv(double *v, double *Xv);
	void XTv(double *v, double *XTv);

	double *C;
	double *z;
	double *D;
	const problem *prob;
};
#endif //HAVE_LAPACK
#endif //_LIBLINEAR_H
