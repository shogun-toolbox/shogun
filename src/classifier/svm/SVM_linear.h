#ifndef _LIBLINEAR_H
#define _LIBLINEAR_H

#include "classifier/svm/Tron.h"
#include "features/SparseFeatures.h"

#ifdef __cplusplus
extern "C" {
#endif

struct problem
{
	int l, n;
	int *y;
	CSparseFeatures<DREAL>* x;
	bool use_bias;
};

struct parameter
{
	int solver_type;

	/* these are for training only */
	double eps;	        /* stopping criteria */
	double C;
	int nr_weight;
	int *weight_label;
	double* weight;
};

struct model
{
	struct parameter param;
	int nr_class;		/* number of classes */
	int nr_feature;
	double *w;
	int *label;		/* label of each class (label[n]) */
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

class l2loss_svm_fun : public function
{
public:
	l2loss_svm_fun(const problem *prob, double Cp, double Cn);
	~l2loss_svm_fun();
	
	double fun(double *w);
	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);

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

class l2_lr_fun : public function
{
public:
	l2_lr_fun(const problem *prob, double Cp, double Cn);
	~l2_lr_fun();
	
	double fun(double *w);
	void grad(double *w, double *g);
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



#endif /* _LIBLINEAR_H */

