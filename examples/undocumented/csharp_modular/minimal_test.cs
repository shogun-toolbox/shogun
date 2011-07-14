using System;

using org.shogun;

public class minimal_test
{
	static minimal_test()
	{
		System.loadLibrary("Kernel");
	}

	static void Main(string[] argv)
	{
		KernelJNI.init_shogun__SWIG_4();
		GaussianKernel x = new GaussianKernel();
		Console.WriteLine(x.get_width());
		KernelJNI.exit_shogun();
	}
}


