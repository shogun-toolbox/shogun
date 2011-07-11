using System;

public class HelloWorld
{
	public static void Main(string[] args)
	{
		Library.init_shogun_with_defaults();
		GaussianKernel k = new GaussianKernel();
		Console.WriteLine(k.get_width());
	}
}
