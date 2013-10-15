#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/DotKernel.h>
#include <shogun/base/init.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <stdio.h>

using namespace shogun;

class CReverseLinearKernel : public CDotKernel
{
public:
    /** default constructor */
    CReverseLinearKernel() : CDotKernel(0)
    {
    }

    /** destructor */
    virtual ~CReverseLinearKernel()
    {
    }

    /** initialize kernel
     *
     * @param l features of left-hand side
     * @param r features of right-hand side
     * @return if initializing was successful
     */
    virtual bool init(CFeatures* l, CFeatures* r)
    {
        CDotKernel::init(l, r);
        return init_normalizer();
    }

    /** load kernel init_data
     *
     * @param src file to load from
     * @return if loading was successful
     */
    virtual bool load_init(FILE* src)
    {
        return false;
    }

    /** save kernel init_data
     *
     * @param dest file to save to
     * @return if saving was successful
     */
    virtual bool save_init(FILE* dest)
    {
        return false;
    }

    /** return what type of kernel we are
     *
     * @return kernel type UNKNOWN (as it is not part
     *			officially part of shogun)
     */
    virtual EKernelType get_kernel_type()
    {
        return K_UNKNOWN;
    }

    /** return the kernel's name
     *
     * @return name "Reverse Linear"
     */
    inline virtual const char* get_name() const
    {
        return "ReverseLinear";
    }

protected:
    /** compute kernel function for features a and b
     * idx_{a,b} denote the index of the feature vectors
     * in the corresponding feature object
     *
     * @param idx_a index a
     * @param idx_b index b
     * @return computed kernel function at indices a,b
     */
    virtual float64_t compute(int32_t idx_a, int32_t idx_b)
    {
        int32_t alen, blen;
        bool afree, bfree;

        float64_t* avec=
            ((CDenseFeatures<float64_t>*) lhs)->get_feature_vector(idx_a, alen, afree);
        float64_t* bvec=
            ((CDenseFeatures<float64_t>*) rhs)->get_feature_vector(idx_b, blen, bfree);

        ASSERT(alen==blen);

        float64_t result=0;
        for (int32_t i=0; i<alen; i++)
            result+=avec[i]*bvec[alen-i-1];

        ((CDenseFeatures<float64_t>*) lhs)->free_feature_vector(avec, idx_a, afree);
        ((CDenseFeatures<float64_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

        return result;
    }
};

void print_message(FILE* target, const char* str)
{
    fprintf(target, "%s", str);
}

int main(int argc, char** argv)
{
    init_shogun(&print_message);

    // create some data
    SGMatrix<float64_t> matrix(2,3);
    for (int32_t i=0; i<6; i++)
        matrix.matrix[i]=i;

    // create three 2-dimensional vectors
    // shogun will now own the matrix created
    CDenseFeatures<float64_t>* features= new CDenseFeatures<float64_t>();
    features->set_feature_matrix(matrix);

    // create reverse linear kernel
    CReverseLinearKernel* kernel = new CReverseLinearKernel();
    kernel->init(features,features);

    // print kernel matrix
    for (int32_t i=0; i<3; i++)
    {
        for (int32_t j=0; j<3; j++)
            SG_SPRINT("%f ", kernel->kernel(i,j));

        SG_SPRINT("\n");
    }

    // free up memory
    SG_UNREF(kernel);

    exit_shogun();
    return 0;
}

