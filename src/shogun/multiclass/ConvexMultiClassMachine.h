/* 
 * File:   ConvexMultiClassMachine.h
 * Author: Mohamed Taher Alrefaie
 *
 * Created on April 12, 2012, 12:31 PM
 */

#ifndef _CONVEXMULTICLASSMACHINE_H___
#define	_CONVEXMULTICLASSMACHINE_H___

#include <shogun/lib/config.h>
#ifdef HAVE_LAPACK
#include <shogun/classifier/svm/SVM_linear.h>
#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/v_array.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/machine/LinearMulticlassMachine.h>

namespace shogun {
    /**
     * method : 
     * feat= orthonormal feature learning, i.e. using trace(W' f(D) W)
     * independent= learning with no coupling across tasks (i.e. using ||W||_2 regularization)
     * diagonal= variable (feature) selection (i.e. D is diagonal)
     */
    #ifndef DOXYGEN_SHOULD_SKIP_THIS
    enum TRAINING_METHOD
    {
        FEAT=0,
        INDEPENDENT=1,
        DIAGONAL=2
    };
    /** Type of min_d evaluation method 
     * it's a function for minimizing over D of the form 
     * called by train_alternating() fn.
     */
    enum D_MIN_METHOD
    {
        d_min,
        d_min_e
    };
    #endif
/**
 * Main algorithm for Multi-task Feature Learning (with a linear kernel)
 * See [Argyriou,Evgeniou,Pontil, NIPS 2006, ML journal 2007]
 */
class CConvexMultiClassMachine : public CLinearMulticlassMachine 
{

public:
    /*default constructor*/
    CConvexMultiClassMachine();
       
    /**
     * 
     * @param features features
     * @param labs labels
     * @param method the method used to train the kernel
     * @param gammas vector of gammas:regularization parameter
     * @param d_ini 
     * @param init_epsilon
     * @param kernel type of kernel used e.g. SVM, least square, regression etc.)
     * @param iterations number of iterations performed
     * @param task_indexes sizes of samples per task (may be unbalanced)
     */
    CConvexMultiClassMachine(CDotFeatures* features, CLabels* labs, TRAINING_METHOD method, SGVector<float64_t> gammas,SGMatrix<float64_t> d_ini,
    float64_t init_epsilon, CKernel* kernel, int32_t iterations, SGVector<int32_t> task_indexes);
    
    /*destructor*/
    virtual ~CConvexMultiClassMachine();
    
    SGMatrix<float64_t> get_d_ini() const {
    return m_d_ini;
}

    void set_d_ini(SGMatrix<float64_t> d_ini_) {
        this->m_d_ini = d_ini_;
    }

    SGVector<float64_t> get_gammas() const {
        return m_gammas;
    }

    void set_gammas(SGVector<float64_t> gammas_) {
        this->m_gammas = gammas_;
    }

    float64_t get_init_epsilon() const {
        return m_epsilon_init;
    }

    void set_init_epsilon(float64_t init_epsilon_) {
        this->m_epsilon_init = init_epsilon_;
    }

    int32_t get_iterations() const {
        return m_iterations;
    }

    void set_iterations(int32_t iterations_) {
        this->m_iterations = iterations_;
    }

    CKernel* get_kernel() const {
        return m_kernel;
    }

    void set_kernel(CKernel* kernel_) {
        this->m_kernel = kernel_;
    }

    TRAINING_METHOD get_method() const {
        return m_method;
    }

    void set_method(TRAINING_METHOD method_) {
        this->m_method = method_;
    }

    SGVector<int32_t> get_task_indexes() const {
        return m_task_indexes;
    }

    void set_task_indexes(SGVector<int32_t> task_indexes_) {
        this->m_task_indexes = task_indexes_;
    }

    
    /** get name */
    virtual const char* get_name() const
    {
            return "CConvexMultiClassMachine";
    }
protected:
    /** train machine */
    virtual bool train_machine(CFeatures* data = NULL);


private:
    /** Initialize parameters*/
    void init_defaults();
    /** Register parameters */
    void register_parameters();
    /**
     * 
     * @param matrix
     */
    /**evaluates (computes) f(D) (acts on the singular values of D)*/
    void f_method(float64_t* vec, float64_t* new_vec, int32_t len);
    /**method for minimizing over D of the form 
     * min_d { sum_i f(d_i) b_i^2 } 
     *(b_i are the singular values of W, or in case of var. selection the L2 norms of the rows of W)*/
    void d_method(float64_t* vec, float64_t* new_vec, int32_t len);
    void train_alternating_epsilon(SGMatrix<float64_t> out_w, SGMatrix<float64_t> out_d, float64_t* out_costfunc, float64_t* out_mineps);
    void train_alternating(SGMatrix<float64_t> out_w, SGMatrix<float64_t> out_d, float64_t* out_costfunc);
    void d_min_e_method(float64_t* vec, float64_t* new_vec, int32_t len);
    void test_error_unbalanced(CDotFeatures* features_test, SGMatrix<float64_t> estimated_w, SGVector<int32_t> task_indexes_test, float64_t* testerr);

    
protected:
    
    TRAINING_METHOD m_method;
    SGVector<float64_t> m_gammas;
    SGMatrix<float64_t> m_d_ini;
    float64_t m_epsilon_init;
    CKernel* m_kernel;
    int32_t m_iterations;
    SGVector<int32_t> m_task_indexes;
    TRAINING_METHOD training_method;
    D_MIN_METHOD d_min_method;
    
    
    

}; //end of class

}





#endif	/* HAVE_LAPACK */
#endif 
