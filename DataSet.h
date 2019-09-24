#pragma once
#ifndef DATASET_H_INCLUDED
#define DATASET_H_INCLUDED
#include <fstream>
#include <vector>
#include <conio.h>
#include <thread>
#include "ConvFeedForward.h"
#include "Volume.h"
#include "Matrix.h"
#include "ConvFeedForward.h"
#include "NN_Tools.h"
typedef matrix<float> Matrix;
///////////////////////////////////////////////////////////////////////////////
//////////////////////////GET DATASET FROM HARD DISK///////////////////////////
///////////////////////////////////////////////////////////////////////////////
void InputOutput(matrix<float>& x,matrix<float>& y, string ET);
int LittleEndian(uint32_t ii);
void get_dataset(Matrix* X,Matrix* Y,const char*Xdir,const char*Ydir,uint32_t EXAMPLES);
void Shuffle(Matrix* X, Matrix* Y);
int get_dataset_2D(Volume& X, Matrix* Y,const char*Xdir, const char*Ydir,uint32_t EXAMPLES);
void Shuffle(Volume& X, Matrix* Y);
void SWAP(Matrix* MAT, int i, int k);
void SWAP(Volume& Vol, int i, int k);
void DevSet(Matrix* X, Matrix* Y, Matrix* X_dev, Matrix* Y_dev, int DEV);
void normalize(Matrix& X, Matrix& X_test, Matrix& Y, Matrix& Y_test);
///////////////////////////////////////////////////////////////////////////////
//////////////////////////ELASTIC DISTORTION///////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
Matrix* gausianFilter(int x, int y, float sigma);
Matrix* gaussianBlur(Matrix* img, int filterSize, float sigma);
float norm_L1(Matrix* x);
Matrix* elasticDistortion(Matrix* img, int filterSize, float sigma, float alpha);
void enlarge2D(IntMatrix* X_1D, Volume& X, int SIZE, int enlargeFact, int threadnum);
IntMatrix* enlarge1D(Volume& X, Matrix*& Y, int enlargeFact);
void to_1D(IntMatrix* X_new, Matrix* X, int colNum);
///////////////////////////////////////////////////////////////////////////////
/////////////////////////END END END END END///////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

#endif // DATASET_H_INCLUDED
