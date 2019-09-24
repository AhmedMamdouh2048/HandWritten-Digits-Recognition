#pragma once
#ifndef VECTVOLUME_HEADER
#define VECTVOLUME_HEADER
#include "Volume.h"
class VectVolume
{
private:
	vector<Volume>* v;
	int numOfVolumes;
public:
	VectVolume();
	VectVolume(int n);
	VectVolume(int n, int channels);
	VectVolume(int n, int channels, int rows, int columns);
	VectVolume(int n, int channels, int rows, int columns, MatrixType Type);
	~VectVolume();
	void DELETE();
	Volume& operator [](int i);
	void resize(int n);
	void pushVol(Volume Mat);
	int size();
	void print();

};
#endif // !VECTVOLUME_HEADER


