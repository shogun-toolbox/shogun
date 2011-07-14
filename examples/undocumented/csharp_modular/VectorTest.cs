using System;

using org.shogun;
using org.jblas;
public class VectorTest
{
	static VectorTest()
	{
		System.loadLibrary("Features");
	}

	static void Main(string[] argv)
	{
		Features.init_shogun_with_defaults();

		double[][] y = { new double[] { 1, 2, 3, 4 } };
		DoubleMatrix A = new DoubleMatrix(y);
		Labels x = new Labels(A);
		DoubleMatrix B = x.get_labels();
		Console.WriteLine(B.ToString());
		Features.exit_shogun();
	}
}
