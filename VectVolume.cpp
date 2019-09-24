#include "VectVolume.h"
////////////////////////////////////
VectVolume::VectVolume()
{
	v = NULL;
	numOfVolumes = 0;
}
////////////////////////////////////
VectVolume::VectVolume(int n)
{
	numOfVolumes = n;
	v = new vector<Volume>(n);
}
////////////////////////////////////
VectVolume::VectVolume(int n, int channels)
{
	numOfVolumes = n;
	v = new vector<Volume>(n);
	for (int i = 0; i < numOfVolumes; i++)
	{
		(*v)[i].resize(channels);
	}
}
////////////////////////////////////
VectVolume::VectVolume(int n, int channels, int rows, int columns)
{
	numOfVolumes = n;
	v = new vector<Volume>(n);
	for (int i = 0; i < numOfVolumes; i++)
	{
		(*v)[i].resize(channels);
		for (int j = 0; j < channels; j++)
		{
			(*v)[i][j] = new Matrix(rows, columns, 0);
		}
	}
}
////////////////////////////////////
VectVolume::VectVolume(int n, int channels, int rows, int columns, MatrixType Type)
{
	numOfVolumes = n;
	v = new vector<Volume>(n);
	for (int i = 0; i < n; i++)
	{
		(*v)[i].resize(channels);
		for (int j = 0; j < channels; j++)
		{
			(*v)[i][j] = new Matrix(rows, columns, Type);
		}
	}
}
////////////////////////////////////
VectVolume::~VectVolume()
{}
////////////////////////////////////
void VectVolume::DELETE()
{
	for (int i = 0; i < numOfVolumes; i++)
		(*v)[i].DELETE();
	delete v;
}
////////////////////////////////////
void VectVolume::pushVol(Volume Vol)
{
	v->push_back(Vol);
	numOfVolumes++;
}
////////////////////////////////////
Volume& VectVolume::operator[](int i)
{
	return (*v)[i];
}
////////////////////////////////////
void VectVolume::resize(int n)
{
	numOfVolumes = n;
	v = new vector<Volume>(n);
}
////////////////////////////////////
int VectVolume::size()
{
	return numOfVolumes;
}
////////////////////////////////////
void VectVolume::print()
{
	cout << "no. 3D Volumes = " << numOfVolumes << endl;
	for (int i = 0; i < numOfVolumes; i++)
	{
		cout << endl;
		(*v)[i].print();
		cout << endl;
	}
}
////////////////////////////////////
