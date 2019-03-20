//The following code can be debuged by Microsoft Visual Studio 2017 with default settings.
//This is a very simple convolution neural network specially designed for image recongnition.
//For my lacking knowledge in derivation for multi-valuable function, there is only one back propagation in this alglorithm.
//In the beginning of the project, I write down the function softmax, which is used for back propagation.
//I have no kowledge back then, but I know if I do not construct it, I will never ever try to do back propagation.
//There is a mistake in the program: I set input[3] for RBG channels, but after the fisrt convolution, they are set to one dimension. The reason why I write the program in this way is due to the overflow of the memory.
//I PROMISE THIS PROGRAM IS GONG TO BE FUN!
//copyright: Yanchong Zheng
#include "pch.h"
#include <iostream>
#include <cmath>
#include<string>
#include <ctime>//generating random numbers for initialing
using namespace std;
float *softmax(float so_in[9],int count);
class CNN
{
private:
	float input[20][20][3];//define an input array to receive the pre-treated data and assume that the input array is 3D.
	float core1[6][3][3][3];//define default convolutional cores.
	float core2[6][3][3][3];//second convolution filter
	float core3[8][3][3][3];//filters
	float cona[6][18][18];//first convolution
	float conb[6][7][7];//second convolution
	float pool1[3][9][9];//first pooling
	float pool2[3][7][7];//second pooling
	float w1[8][5][5];//weight 1
	float w2[200][10][4];//weight 2
	float w3[10][10][4];//weight 3
	float w4[10][10][4];//weight 4
	float output1[8][5][5];//first output
	float output2[200][4];//output1 here becomes a 4D vector.
	short int a = 0;
	float conres1[8][5][5];//convolution result
	float pro1[10];//I use drop-out nodes to reduce the probility of over-fitting
	float pro2[10];//drop-out nodes again
	float pro3[10];//drop-out nodes again
	float fr1[10][4];//first connect
	float fr2[10][4];//second connect 
	float fr3[10][4];//third connect   
	float object[9][4];
	float min_d = d[0];
	short int count = 0;
	float d[9];
	float *so_res;
	float random()
	{
		return rand() / (float)RAND_MAX;
	}
public:
	int initi()
	{
		std::cout << "Initializing a Random Cnovolution Neural Network..." << endl;
		srand(time(NULL));//setting time as seed
		return 0;
	}
	int initip()
	{
		for (int i = 0; i < 20; i++)//initialing input
		{
			for (int j = 0; j < 20; j++)
			{
				for (int k = 0; k < 3; k++)
				{
					input[i][j][k] = rand() % 128 - 64;
				}
			}
		}
		return 0;
	}
	int initf()
	{
		std::cout << "Initializing Random Cnovolution Neural Network filters..." << endl;
		for (int m = 0; m < 6; m++)//initialing filters
		{
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					for (int k = 0; k < 3; k++)
					{
						core1[m][i][j][k] = random() - 0.5;
					}
				}
			}
		}
		for (int m = 0; m < 6; m++)//initialing filters
		{
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					for (int k = 0; k < 3; k++)
					{
						core2[m][i][j][k] = random() - 0.5;
					}
				}
			}
		}
		for (int m = 0; m < 8; m++)//initialing filters
		{
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					for (int k = 0; k < 3; k++)
					{
						core3[m][i][j][k] = random() - 0.5;
					}
				}
			}
		}
		std::cout << "Initializing Random Cnovolution Neural Network filters...DONE" << endl;
		return 0;
	}
	int conPool()
	{
			for (int h = 0; h < 6; h++)//first convolution
			{
				for (int i = 0; i < 18; i++)
				{
					for (int j = 0; j < 18; j++)
					{
						cona[h][i][j] = 0;
						for (int l = i; l < i + 3; l++)
						{
							for (int m = j; m < j + 3; m++)
							{
								for (int n = 0; n < 3; n++)
								{
									cona[h][i][j] = cona[h][i][j] + input[l][m][n] * core1[h][l - i][j - m][n];//set bias 
								}
							}
						}
						cona[h][i][j] = cona[h][i][j] / 3.0;
						if (cona[h][i][j] < 0) cona[h][i][j] = 0;//ReLU function
					}
				}
			}
		for (int i = 0; i < 3; i++)//first maxpooling
		{
			for (int j = 0; j < 9; j++)
			{
				for (int k = 0; k < 9; k++)
				{
					pool1[i][j][k] = cona[2 * i][2 * j][2 * k];
					if (cona[2 * i][2 * j + 1][2 * k] > pool1[i][j][k]) pool1[i][j][k] = cona[2 * i][2 * j + 1][2 * k];
					if (cona[2 * i][2 * j + 1][2 * k + 1] > pool1[i][j][k]) pool1[i][j][k] = cona[2 * i][2 * j + 1][2 * k + 1];
					if (cona[2 * i][2 * j][2 * k + 1] > pool1[i][j][k]) pool1[i][j][k] = cona[2 * i][2 * j][2 * k + 1];
					if (cona[2 * i + 1][2 * j][2 * k + 1] > pool1[i][j][k]) pool1[i][j][k] = cona[2 * i + 1][2 * j][2 * k + 1];
					if (cona[2 * i + 1][2 * j][2 * k + 1] > pool1[i][j][k]) pool1[i][j][k] = cona[2 * i + 1][2 * j][2 * k + 1];
					if (cona[2 * i + 1][2 * j][2 * k + 1] > pool1[i][j][k]) pool1[i][j][k] = cona[2 * i + 1][2 * j][2 * k + 1];
				}
			}
		}//9-9-3
		for (int h = 0; h < 6; h++)//second convolution
		{
			for (int i = 0; i < 7; i++)
			{
				for (int j = 0; j < 7; j++)
				{
					conb[h][i][j] = 0;
					for (int l = i; l < i + 3; l++)
					{
						for (int m = j; m < j + 3; m++)
						{
							for (int n = 0; n < 3; n++)
							{
								conb[h][i][j] = conb[h][i][j] + pool1[n][l][m] * core2[h][l - i][m - j][n];
							}
						}
					}
					conb[h][i][j] = conb[h][i][j] / 3.0;
					if (conb[h][i][j] < 0) conb[h][i][j] = 0;//ReLU
				}
			}
		}//7-7-6
		for (int i = 0; i < 3; i++)//second pooling
		{
			for (int j = 0; j < 7; j++)
			{
				for (int k = 0; k < 7; k++)
				{
					pool2[i][j][k] = conb[2 * i][j][k];
					if (conb[2 * i][1][k] > pool2[i][j][k]) pool2[i][j][k] = conb[2 * i][j][k];
					if (conb[2 * i][j][2 * k + 1] > pool2[i][j][k]) pool2[i][j][k] = conb[2 * i][j][k];
					if (conb[2 * i][j][2 * k + 1] > pool2[i][j][k]) pool2[i][j][k] = conb[2 * i][j][k];
					if (conb[2 * i + 1][j][2 * k + 1] > pool2[i][j][k]) pool2[i][j][k] = conb[2 * i + 1][j][k];
					if (conb[2 * i + 1][j][2 * k + 1] > pool2[i][j][k]) pool2[i][j][k] = conb[2 * i + 1][j][k];
					if (conb[2 * i + 1][j][2 * k + 1] > pool2[i][j][k]) pool2[i][j][k] = conb[2 * i + 1][j][k];
				}
			}
		}//7-7-3
		for (int h = 0; h < 8; h++)//third covolution
		{
			for (int i = 0; i < 5; i++)
			{
				for (int j = 0; j < 5; j++)
				{
					conres1[h][i][j] = 0;
					for (int l = i; l < i + 3; l++)
					{
						for (int m = j; m < j + 3; m++)
						{
							for (int n = 0; n < 3; n++)
							{
								conres1[h][i][j] = conres1[h][i][j] + pool2[n][l][m] * core3[h][l - i][m - j][n];//set bias 
							}
						}
					}
					if (conres1[h][i][j] < 0) conres1[h][i][j] = 0;//ReLU
					conres1[h][i][j] = conres1[h][i][j] / 3.0;
				}
			}
		}//5-5-8
		return 0;
	}
	int initwp()
	{
		std::cout << "Initializing weight and prosibilities..." << endl;
		for (int i = 0; i < 8; i++)//Initializing weight1
		{
			for (int j = 0; j < 5; j++)
			{
				for (int k = 0; k < 5; k++)
				{
					w1[i][j][k] = (rand() % 10) / 10.0;
				}
			}
		}
		for (int i = 0; i < 200; i++)//Initializing weight 2
		{
			for (int j = 0; j < 10; j++)
			{
				for (int k = 0; k < 4; k++)
				{
					w2[i][j][k] = (rand() % 10) / 1000.0;
				}
			}
		}
		for (int i = 0; i < 10; i++)//Initializing
		{
			pro1[i] = (rand() % 10) / 10.0;
			if (pro1[i] < 0.3) pro1[i] = 0;
			if (pro1[i] >= 0.3) pro1[i] = 1;
		}
		for (int i = 0; i < 10; i++)//Initializing
		{
			for (int j = 0; j < 10; j++)
			{
				for (int k = 0; k < 4; k++)
				{
					w3[i][j][k] = (rand() % 10) / 100.0;
				}
			}
		}
		for (int i = 0; i < 10; i++)
		{
			pro2[i] = (rand() % 10) / 10.0;
			if (pro2[i] < 0.3) pro2[i] = 0;
			if (pro2[i] >= 0.3) pro2[i] = 1;
		}
		for (int i = 0; i < 10; i++)
		{
			for (int j = 0; j < 10; j++)
			{
				for (int k = 0; k < 4; k++)
				{
					w4[i][j][k] = (rand() % 10) / 100.0;
				}
			}
		}
		for (int i = 0; i < 10; i++)
		{
			pro3[i] = (rand() % 10) / 10.0;
			if (pro3[i] < 0.3) pro3[i] = 0;
			if (pro3[i] >= 0.3) pro3[i] = 1;
		}
		std::cout << "Initializing weight and prosibilities...DONE" << endl;
		return 0;
	}
	int fconnect()
	{
		for (int i = 0; i < 8; i++)//calculating output
		{
			for (int j = 0; j < 5; j++)
			{
				for (int k = 0; k < 5; k++)
				{
					output1[i][j][k] = w1[i][j][k] * conres1[i][j][k];
				}
			}
		}
		for (int j = 0; j < 8; j++)//dimension reduction
		{
			for (int k = 0; k < 5; k++)
			{
				for (int l = 0; l < 5; l++)
				{
					output2[a][0] = j;
					output2[a][1] = k;
					output2[a][2] = l;
					output2[a][3] = output1[j][k][l];
					a++;
				}
			}
		}
		for (int i = 0; i < 10; i++)//calculating
		{
			for (int j = 0; j < 4; j++)
			{
				fr1[i][j] = 0;
				if (pro1[i] == 1)
				{
					for (int k = 0; k < 200; k++)
					{
						fr1[i][j] = fr1[i][j] + w2[k][i][j] * output2[k][j];
					}
				}
			}
		}
		for (int i = 0; i < 10; i++)//second connect
		{
			for (int j = 0; j < 4; j++)
			{
				fr2[i][j] = 0;
				if (pro2[i] == 1)
				{
					for (int k = 0; k < 10; k++)
					{
						fr2[i][j] = fr2[i][j] + w3[k][i][j] * fr1[k][j];
					}
				}
			}
		}
		for (int i = 0; i < 10; i++)//calculating
		{
			for (int j = 0; j < 4; j++)
			{
				fr3[i][j] = 0;
				if (pro3[i] == 1)
				{
					for (int k = 0; k < 10; k++)
					{
						fr3[i][j] = fr3[i][j] + w4[k][i][j] * fr2[k][j];
					}
				}
				fr3[i][j] = fr3[i][j] -0.1;
			}
		}
		return 0;
	}
	int inito()
	{
		std::cout << "Initializing Objects' Features" << endl;
		for (int i = 0; i < 9; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				object[i][j] = random()-0.5;
			}
			object[i][3] = random()*5.0;
		}
		std::cout << "Initializing Objects' Features...DONE" << endl;
		return 0;
	}
	int match()
	{
		for (int i = 0; i < 9; i++)
		{
			d[i] = 0;
			for (int j = 0; j < 10; j++)
			{
				for (int k = 0; k < 4; k++)
				{
					d[i] = d[i] + (fr3[j][k] - object[i][k])*(fr3[j][k] - object[i][k])*(fr3[j][k] - object[i][k]);
				}
			}
			d[i] = d[i] / 4000.0;
		}
		for (int i = 0; i < 9; i++)
		{
			if (min_d > d[i])
			{
				min_d = d[i];
				count = i;
			}
		}
		so_res = softmax(d,count);//SoftMax Function
		return count;
	}
	int updatew4()
	{
		for (int i = 0;i < 9; i++)
		{
			for (int j = 0; j < 10; j++)
			{
				for (int k = 0; k < 10; k++)
				{
					for (int l = 0; l < 4; l++)
					{
						w4[j][k][l] = w4[j][k][l] - so_res[i] / 100.0+0.0011111205;
					}
				}
			}
		}
		return 0;
	}
	int debug_input()
	{
		for (int i = 0; i < 20; i++)
		{
			for (int j = 0; j < 20; j++)
			{
				for (int k = 0; k < 3; k++)
				{
					std::cout << input[i][j][k] << endl;
				}
			}
		}
		return 0;
	}
	int debug_core1()
	{
		for (int m = 0; m < 6; m++)
		{
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					for (int k = 0; k < 3; k++)
					{
						std::cout << core1[m][i][j][k] << endl;
					}
				}
			}
		}
		return 0;
	}
	int debug_cona()
	{
		for (int h = 0; h < 6; h++)
		{
			for (int i = 0; i < 18; i++)
			{
				for (int j = 0; j < 18; j++)
				{
					std::cout << cona[h][i][j] << endl;
				}
			}
		}
		return 0;
	}
	int debug_pool1()
	{
		for (int i = 0; i < 3; i++)//first maxpooling
		{
			for (int j = 0; j < 9; j++)
			{
				for (int k = 0; k < 9; k++)
				{
					std::cout << pool1[i][j][k] << endl;
				}
			}
		}//9-9-3
		return 0;
	}
	int debug_conb()
	{
		for (int h = 0; h < 6; h++)
		{
			for (int i = 0; i < 7; i++)
			{
				for (int j = 0; j < 7; j++)
				{
					std::cout << conb[h][i][j] << endl;
				}
			}
		}//7-7-6
		return 0;
	}
	int debug_pool2()
	{
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 7; j++)
			{
				for (int k = 0; k < 7; k++)
				{
					std::cout << pool2[i][j][k] << endl;
				}
			}
		}//7-7-3
		return 0;
	}
	int debug_conres()
	{
		for (int h = 0; h < 8; h++)//third covolution
		{
			for (int i = 0; i < 5; i++)
			{
				for (int j = 0; j < 5; j++)
				{
					std::cout << conres1[h][i][j] << endl;
				}
			}
		}//5-5-8
		return 0;
	}
	int debug_pro1()
	{
		for (int i = 0; i < 10; i++)//Initializing
		{
			std::cout << pro1[i] << endl;
		}
		return 0;
	}
	int debug_w1()
	{
		for (int i = 0; i < 8; i++)
		{
			for (int j = 0; j < 5; j++)
			{
				for (int k = 0; k < 5; k++)
				{
					std::cout << w1[i][j][k] << endl;
				}
			}
		}
		return 0;
	}
	int debug_output1()
	{
		for (int i = 0; i < 8; i++)
		{
			for (int j = 0; j < 5; j++)
			{
				for (int k = 0; k < 5; k++)
				{
					std::cout << output1[i][j][k] << endl;
				}
			}
		}
		return 0;
	}
	int debug_output2()
	{
		for (int j = 0; j < 200; j++)
		{
			for (int k = 0; k < 4; k++)
			{
				std::cout << output2[j][k] << endl;
			}
		}
		return 0;
	}
	int debug_fr1()
	{
		for (int i = 0; i < 10; i++)//calculating
		{
			for (int j = 0; j < 4; j++)
			{
				std::cout << fr1[i][j] << endl;
			}
		}
		return 0;
	}
	int debug_fr2()
	{
		for (int i = 0; i < 10; i++)//calculating
		{
			for (int j = 0; j < 4; j++)
			{
				std::cout << fr2[i][j] << endl;
			}
		}
		return 0;
	}
	int debug_fr3()
	{
		for (int i = 0; i < 10; i++)//calculating
		{
			for (int j = 0; j < 4; j++)
			{
				std::cout << fr3[i][j] << endl;
			}
		}
		return 0;
	}
	int debug_object()
	{
		for (int i = 0; i < 9; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				std::cout << object[i][j] << endl;
			}
		}
		return 0;
	}
	int debug_d()
	{
		for (int i = 0; i < 9; i++)
		{
			std::cout << d[i] << endl;
		}
		return 0;
	}
};
class Cat
{
private:
	string name;
public:
	void ask()
	{
		string recname;
		string boolean;
		bool b = true;
		std::cout << "Hello" << endl;
		while (b)
		{
			std::cout << "Would you like to give me a first name?" << endl;
			std::cout << "Type yes or no..." << endl;
			getline(cin, boolean);
			if (boolean == "yes")
			{
				std::cout << "Now write it down..." << endl;
				getline(cin, recname);
				name = recname;
				b = false;
			}
			else if (boolean == "no")
			{
				std::cout << "I am feeling a little sad, but it is okay. You can call Tom." << endl;
				b = false;
			}
			else
				std::cout << "Invalid Input. Please give me an answer as the instruction." << endl;
		}
	}
	string getname()
	{
		return name;
	}
};
class Dog
{
private:
	string name;
public:
	void ask()
	{
		string recname;
		string boolean;
		bool b = true;
		std::cout << "Hello" << endl;
		while (b)
		{
			std::cout << "Would you like to give me a first name?" << endl;
			std::cout << "Type yes or no..." << endl;
			getline(cin, boolean);
			if (boolean == "yes")
			{
				std::cout << "Now write it down..." << endl;
				getline(cin, recname);
				name = recname;
				b = false;
			}
			else if (boolean == "no")
			{
				std::cout << "I am feeling a little sad, but it is okay. You can call Jerry." << endl;
				b = false;
			}
			else
				std::cout << "Invalid Input. Please give me an answer as the instruction." << endl;
		}
	}
	string getname()
	{
		return name;
	}
};
class Bird
{
private:
	string name;
public:
	void ask()
	{
		string recname;
		string boolean;
		bool b = true;
		std::cout << "Hello" << endl;
		while (b)
		{
			std::cout << "Would you like to give me a first name?" << endl;
			std::cout << "Type yes or no..." << endl;
			getline(cin, boolean);
			if (boolean == "yes")
			{
				std::cout << "Now write it down..." << endl;
				getline(cin, recname);
				name = recname;
				b = false;
			}
			else if (boolean == "no")
			{
				std::cout << "I am feeling a little sad, but it is okay. You can call Alan." << endl;
				b = false;
			}
			else
				std::cout << "Invalid Input. Please give me an answer as the instruction." << endl;
		}
	}
	string getname()
	{
		return name;
	}
};
class Bailey : public Cat
{
public:
	void hello()
	{
		std::cout << "Hello, I am a Bailey Cat, and Bailey is my last name." << endl;
	}
	void paint()
	{
		std::cout << " ∧, ,,∧  " << endl;
		std::cout << "/●   ● \\" << endl;
		std::cout << "╰/] - [\ " << endl;
	};
};
class Bombalurina : public Cat
{
public:
	void hello()
	{
		std::cout << "Hello, I am a Bombalurina Cat, and Bombalurina is my last name." << endl;
	}
	void paint()
	{
		std::cout << "’>’" << endl;
	};
};
class Asparagus : public Cat
{
public:
	void hello()
	{
		std::cout << "Hello, I am a Asparagus Cat, and Asparagus is my last name." << endl;
	}
	void paint()
	{
		std::cout << "(* : *)" << endl;
	};
};
class Carbucketty : public Dog
{
public:
	void hello()
	{
		std::cout << "Hello, I am a Carbucketty Dog, and Carbucketty is my last name." << endl;
	}
	void paint()
	{
		std::cout << "(＾_＾)" << endl;
	};
};
class Cassandra : public Dog
{
public:
	void hello()
	{
		std::cout << "Hello, I am a Cassandra Dog, and Cassandra is my last name." << endl;
	}
	void paint()
	{
		std::cout << "(￣  -￣)" << endl;
	};
};
class Coricopat : public Dog
{
public:
	void hello()
	{
		std::cout << "Hello, I am a Coricopat Dog, and Coricopat is my last name." << endl;
	}
	void paint()
	{
		std::cout << "(* ..*)" << endl;
	};
};
class Demeter : public Bird
{
public:
	void hello()
	{
		std::cout << "Hello, I am a Demeter Bird, and Demeter is my last name." << endl;
	}
	void paint()
	{
		std::cout << "(￣ ;￣)" << endl;
	};
};
class Electra : public Bird
{
public:
	void hello()
	{
		std::cout << "Hello, I am a Electra Bird, and Electra is my last name." << endl;
	}
	void paint()
	{
		std::cout << "( ^*^)" << endl;
	};
};
class Exotica : public Bird
{
public:
	void hello()
	{
		std::cout << "Hello, I am a Exotica Bird, and Exotica is my last name." << endl;
	}
	void paint()
	{
		std::cout << "(* & *)" << endl;
	};
};
int main()
{
	std::cout << "Hello!" << endl;
	std::cout << "Welcome to the world of Convolution Neural Network!" << endl;
	std::cout << "I promise it is going to be Fun!" << endl;
	CNN C;
	C.initi();
	C.initip();
	C.initf();
	C.conPool();
	C.initwp();
	C.fconnect();
	C.inito();
	int count;
	count = C.match();
	C.updatew4();
	std::cout << "Initializing Cnovolution Neural Network...DONE" << endl;
	std::cout << "Start training..." << endl;
	for (int i = 0; i < 10000; i++)
	{
		C.initip();;
		C.conPool();
		C.fconnect();
		count = C.match();
		C.updatew4();
	}
	std::cout << "Training...Done" << endl;
	std::cout << "Initializing Random Cnovolution Neural Network inputs..." << endl;
	C.initip();
	std::cout << "Initializing Random Cnovolution Neural Network inputs...DONE" << endl;
	C.initf();
	std::cout << "Start Convolution and Pooling ..." << endl;
	C.conPool();
	std::cout << "Convolution and Pooling Done..." << endl;
	C.initwp();
	std::cout << "Start connecting Data ..." << endl;
	C.fconnect();
	std::cout << "Data Connected." << endl;
	C.inito();
	std::cout << "Start matching..." << endl;
	count = C.match();
	std::cout << "Matching...DONE" << endl;
	std::cout << "Convoluion Neural Network shuts down." << endl;
	std::cout << "I am coming out!" << endl;
	std::cout << "" << endl;
	if (count == 0)
	{
		Bailey a;
		a.paint();
		std::cout << "" << endl;
		a.hello();
		a.ask();
		string b = a.getname();
		if (b != "") std::cout << b + " is a nice name!" << endl;
		else std::cout << "I don't like blank space..." << endl;
	}
	if (count == 1)
	{
		Bombalurina a;
		a.paint();
		std::cout << "" << endl;
		a.hello();
		a.ask();
		string b = a.getname();
		if (b != "") std::cout << b + " is a nice name!" << endl;
		else std::cout << "I don't like blank space..." << endl;
	}
	if (count == 2)
	{
		Asparagus a;
		a.paint();
		std::cout << "" << endl;
		a.hello();
		a.ask();
		string b = a.getname();
		if (b != "") std::cout << b + " is a nice name!" << endl;
		else std::cout << "I don't like blank space..." << endl;
	}
	if (count == 3)
	{
		Carbucketty a;
		a.paint();
		std::cout << "" << endl;
		a.hello();
		a.ask();
		string b = a.getname();
		if (b != "") std::cout << b + " is a nice name!" << endl;
		else std::cout << "I don't like blank space..." << endl;
	}
	if (count == 4)
	{
		Cassandra a;
		a.paint();
		std::cout << "" << endl;
		a.hello();
		a.ask();
		string b = a.getname();
		if (b != "") std::cout << b + " is a nice name!" << endl;
		else std::cout << "I don't like blank space..." << endl;
	}
	if (count == 5)
	{
		Coricopat a;
		a.paint();
		std::cout << "" << endl;
		a.hello();
		a.ask();
		string b = a.getname();
		if (b != "") std::cout << b + " is a nice name!" << endl;
		else std::cout << "I don't like blank space..." << endl;
	}
	if (count == 6)
	{
		Demeter a;
		a.paint();
		std::cout << "" << endl;
		a.hello();
		a.ask();
		string b = a.getname();
		if (b != "") std::cout << b + " is a nice name!" << endl;
		else std::cout << "I don't like blank space..." << endl;
	}
	if (count == 7)
	{
		Electra a;
		a.paint();
		std::cout << "" << endl;
		a.hello();
		a.ask();
		string b = a.getname();
		if (b != "") std::cout << b + " is a nice name!" << endl;
		else std::cout << "I don't like blank space..." << endl;
	}
	if (count == 8)
	{
		Exotica a;
		a.paint();
		std::cout << "" << endl;
		a.hello();
		a.ask();
		string b = a.getname();
		if (b != "") std::cout << b + " is a nice name!" << endl;
		else std::cout << "I don't like blank space..." << endl;
	}
	std::cout << "I am going to sleep now. When I wake up, I will forget everything." << endl;
	std::cout << "Before I go to sleep, may you tell me your name? " << endl;
	std::cout << "Type yes or no..." << endl;
	bool c = true;
	string yourname, answer;
	while (c)
	{
		getline(cin, answer);
		if (answer == "yes")
		{
			std::cout << "Now write it down..." << endl;
			getline(cin, yourname);
			c = false;
		}
		else if (answer == "no")
		{
			std::cout << "I am feeling a little sad, but it is okay." << endl;
			c = false;
		}
		else
			std::cout << "Invalid Input. Please give me an answer as the instruction." << endl;
	}
	if(yourname!="")std::cout << "Nice to meet you, " + yourname + ". Have a good day! " << endl;
	else std::cout << "You didn't tell me your name. Have a good day! " << endl;
	std::cout << "Bye!" << endl;
	system("pause");
	return 0;

}
float *softmax(float so_in[9],int count)
{
	float sum = 0, a[9];
	for (int i = 0; i < 9; i++)
	{
		a[i] = exp(so_in[i]);
		sum = sum + a[i];
	}
	float *ar = new float[9];
	for (int j = 0; j < 9; j++)
	{
		ar[j] = a[j] / sum;
	}
	a[count] = a[count] - 1;
	return ar;
}
