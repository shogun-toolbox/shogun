%module KernelMachine%{
 #include "kernel/KernelMachine.h"
%}

%include "classifier/Classifier.i"

class CKernelMachine : public CClassifier {
	public:
		CKernelMachine();
		virtual ~CKernelMachine();

		inline void set_kernel(CKernel* k) { kernel=k; }
		inline CKernel* get_kernel() { return kernel; }

	protected:
		CKernel* kernel;
};
