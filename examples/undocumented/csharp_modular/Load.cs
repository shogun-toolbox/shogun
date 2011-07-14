using System;
using System.Collections;

using org.shogun;
using org.jblas;
public class Load
{
	public static DoubleMatrix load_numbers(string filename)
	{
		DoubleMatrix result = null;
		try
		{
			DoubleMatrix temp = DoubleMatrix.loadAsciiFile(filename);
			result = temp.reshape(2, temp.Length / 2);
		}
		catch(java.io.IOException e)
		{
			Console.WriteLine("Unable to create matrix from " + filename + ": " + e.Message);
			Environment.Exit(-1);
		}
		return result;
	}
	public static DoubleMatrix load_labels(string filename)
	{
		DoubleMatrix result = null;
		try
		{
			DoubleMatrix temp = DoubleMatrix.loadAsciiFile(filename);
			result = temp.reshape(1, temp.Length);
		}
		catch(java.io.IOException e)
		{
			Console.WriteLine("Unable to create matrix from " + filename + ": " + e.Message);
			Environment.Exit(-1);
		}
		return result;
	}
	public static string[] load_dna(string filename)
	{
		ArrayList list = new ArrayList();
		string[] result = null;
		try
		{
			FileInputStream fstream = new FileInputStream(filename);
			DataInputStream @in = new DataInputStream(fstream);
			BufferedReader buffer = new BufferedReader(new InputStreamReader(@in));
			string line;
			while((line = buffer.readLine()) != null)
			{
				list.Add(line);
			}
			@in.close();
			result = new string[list.Count];
			for (int i = 0; i < list.Count; i++)
			{
				result[i] = (string)list[i];
			}
		}
		catch(java.io.IOException e)
		{
			Console.WriteLine("Unable to create matrix from " + filename + ": " + e.Message);
			Environment.Exit(-1);
		}
		return result;
	}
}
