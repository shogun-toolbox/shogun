using System;

public class HelloWorld
{
	public static void Main(string[] args)
	{
		Library.init_shogun();
		GaussianKernel k = new GaussianKernel();
		Console.WriteLine(k.get_width());
	}
}
