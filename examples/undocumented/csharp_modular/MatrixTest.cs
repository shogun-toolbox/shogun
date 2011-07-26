using System;

using org.shogun;
using org.jblas;

public class MatrixTest
{
	static MatrixTest()
	{
// The library is specified in the 'DllImport' attribute for .NET:
//		System.loadLibrary("modshogun");
	}

	static void Main(string[] argv)
	{
		modshogun.init_shogun();
		Console.WriteLine("Test DoubleMatrix(jblas):");
		RealFeatures x = new RealFeatures();
		double[][] y = { new double[] { 1, 2 }, new double[] { 3, 4 }, new double[] { 5, 6 } };
		DoubleMatrix A = new DoubleMatrix(y);
		x.set_feature_matrix(A);
		DoubleMatrix B = x.get_feature_matrix();
		Console.WriteLine(B.ToString());
		modshogun.exit_shogun();
	}
}
