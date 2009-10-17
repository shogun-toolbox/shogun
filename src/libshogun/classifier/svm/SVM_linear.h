#ifndef DOXYGEN_SHOULD_SKIP_THIS

#ifndef _LIBLINEAR_H
#define _LIBLINEAR_H

#include "lib/config.h"

#ifdef HAVE_LAPACK
#include "classifier/svm/Tron.h"
#include "features/DotFeatures.h"

namespace shogun
{

#ifdef __cplusplus
extern "C" {
#endif

/** problem */
struct problem
{
	/** l */
	int32_t l;
	/** n */
	int32_t n;
	/** y */
	int32_t *y;
	/** sparse features x */
	CDotFeatures* x;
	/** if bias shall be used */
	bool use_bias;
};

/** parameter */
struct parameter
{
	/** solver type */
	int32_t solver_type;

	/* these are for training only */
	/** stopping criteria */
	float64_t eps;
	/** C */
	float64_t C;
	/** number of weights */
	int32_t nr_weight;
	/** weight label */
	int32_t *weight_label;
	/** weight */
	float64_t* weight;
};

/** model */
struct model
{
	/** parameter */
	struct parameter param;
	/** number of classes */
	int32_t nr_class;
	/** number of features */
	int32_t nr_feature;
	/** w */
	float64_t *w;
	/** label of each class (label[n]) */
	int32_t *label;
	/** bias */
	float64_t bias;
};

struct model* train(const struct problem *prob, const struct parameter *param);
void cross_validation(
	const struct problem *prob, const struct parameter *param, int32_t nr_fold,
	int32_t *target);

int32_t predict_values(
	const struct model *model_, const struct feature_node *x,
	float64_t* dec_values);
int32_t predict(const struct model *model_, const struct feature_node *x);
int32_t predict_probability(
	const struct model *model_, const struct feature_node *x,
	float64_t* prob_estimates);

int32_t save_model(const char *model_file_name, const struct model *model_);
struct model *load_model(const char *model_file_name);

int32_t get_nr_feature(const struct model *model_);
int32_t get_nr_class(const struct model *model_);
void get_labels(const struct model *model_, int32_t* label);

void destroy_model(struct model *model_);
void destroy_param(struct parameter *param);
const char *check_parameter(
	const struct problem *prob, const struct parameter *param);

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
	l2loss_svm_fun(const problem *prob, float64_t Cp, float64_t Cn);
	~l2loss_svm_fun();
	
	/** fun
	 *
	 * @param w w
	 * @return something floaty
	 */
	float64_t fun(float64_t *w);
	
	/** grad
	 *
	 * @param w w
	 * @param g g
	 */
	void grad(float64_t *w, float64_t *g);

	/** Hv
	 *
	 * @param s s
	 * @param Hs Hs
	 */
	void Hv(float64_t *s, float64_t *Hs);

	/** get number of variables
	 *
	 * @return number of variables
	 */
	int32_t get_nr_variable(void);

private:
	void Xv(float64_t *v, float64_t *Xv);
	void subXv(float64_t *v, float64_t *Xv);
	void subXTv(float64_t *v, float64_t *XTv);

	float64_t *C;
	float64_t *z;
	float64_t *D;
	int32_t *I;
	int32_t sizeI;
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
	l2_lr_fun(const problem *prob, float64_t Cp, float64_t Cn);
	~l2_lr_fun();

	/** fun
	 *
	 * @param w w
	 * @return something floaty
	 */
	float64_t fun(float64_t *w);
	
	/** grad
	 *
	 * @param w w
	 * @param g g
	 */
	void grad(float64_t *w, float64_t *g);

	/** Hv
	 *
	 * @param s s
	 * @param Hs Hs
	 */
	void Hv(float64_t *s, float64_t *Hs);

	int32_t get_nr_variable(void);

private:
	void Xv(float64_t *v, float64_t *Xv);
	void XTv(float64_t *v, float64_t *XTv);

	float64_t *C;
	float64_t *z;
	float64_t *D;
	const problem *prob;
};
}
#endif //HAVE_LAPACK
#endif //_LIBLINEAR_H

#endif // DOXYGEN_SHOULD_SKIP_THIS
