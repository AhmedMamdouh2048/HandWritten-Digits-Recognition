#include "Volume.h"
////////////////////////////////////
Volume::Volume()
{
	vol = NULL;
	numOfChannels = 0;
}
////////////////////////////////////
Volume::Volume(int n)
{

	numOfChannels = n;
	vol = new vector<Matrix*>(n);
}
////////////////////////////////////
Volume::Volume(int n, int rows, int columns)
{
	numOfChannels = n;
	vol = new vector<Matrix*>(numOfChannels);
	for (int j = 0; j < numOfChannels; j++)
	{
		(*vol)[j] = new Matrix(rows, columns, 0);
	}
}
////////////////////////////////////
Volume::Volume(int n, int rows, int columns, MatrixType Type)
{
	numOfChannels = n;
	vol = new vector<Matrix*>(numOfChannels);
	for (int j = 0; j < numOfChannels; j++)
	{
		(*vol)[j] = new Matrix(rows, columns, Type);
	}
}
////////////////////////////////////
Volume::~Volume()
{}
////////////////////////////////////
void Volume::DELETE()
{
	for (int j = 0; j < numOfChannels; j++)
	{
		delete (*vol)[j];
	}
    delete vol;
}
////////////////////////////////////
void Volume::pushMat(Matrix* Mat)
{
	vol->push_back(Mat);
	numOfChannels++;
}
////////////////////////////////////
Matrix& Volume::operator()(int i)
{
	return *((*vol)[i]);
}
////////////////////////////////////
Matrix*& Volume::operator[](int i)
{
	return (*vol)[i];
}
////////////////////////////////////
void Volume::resize(int n)
{
	numOfChannels = n;
	vol = new vector<Matrix*>(n);
}
////////////////////////////////////
int Volume::size()
{
	return numOfChannels;
}
////////////////////////////////////
void Volume::print()
{
	cout << "no. 2D matrices = " << numOfChannels << endl;
	for (int i = 0; i < numOfChannels; i++)
	{
		cout << endl;
		(*vol)[i]->print();
		cout << endl;
	}
}
////////////////////////////////////
