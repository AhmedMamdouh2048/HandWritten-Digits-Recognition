#pragma once
#include "VectVolume.h"
typedef matrix<float> Matrix;

Matrix* convolve(Volume& Aprev, Volume& filter, int s);                         // Convolves a volume Aprev with filter
VectVolume convolve(VectVolume Aprev, VectVolume filters, int stride);          // Optimized version of convolve that uses dot product
VectVolume convolve_Threaded(VectVolume Aprev, VectVolume filters, int stride); // Threaded version of convolve with transposed dot product
Matrix* convolve2(Volume& Aprev, Volume& Filter, int s);                        // Uses transposed dot product
void maxPool(Volume& Aprev, Volume& A, int f, int s);                           // Performs max pooling in Aprev and return the result in A. A must be empty volume, its contents is created inside
void avgPool(Volume& Aprev, Volume& A, int f, int s);                           // Performs average pooling in Aprev and return the result in A. A must be empty volume, its contents is created inside
VectVolume to_VectorOfVolume(Matrix* A, int nh, int nw, int nc, int m);         // Converts a Matrix(vector of 1D -flat- images) into a VectVolume(vector of volumes)
Volume	to_2D(Matrix* X);                                                       // Converts a Matrix(vector of 1D -flat- images) into a Volume(vector of 2D images)
Matrix* to_1D(Volume& X_2D);                                                    // Converts a Volume(vector of 2D images) into a Matrix(vector of 1D -flat- images)
Matrix* to_FC(VectVolume A);                                                    // Converts a vector of Volume(output of last conv layer) into a Matrix (noFeatures X m)
Matrix* pad(Matrix* img, int p, float value);                                   // Extends a square matrix into (n+p x n+p) dims, the extended entries have the value value
Matrix* FilterToMatrix(Matrix* filter, int nh, int nw, int s);                  // Converts an fxf filter to an extended matrix of dimensions [ (nh - f + 1) * (nw - f + 1) ] x [ nh * nw ]
Matrix* to_Mat(Matrix* X);                                                      // Converts a 1D falattened vector into 2D matrix
Volume Imgs2Vects(VectVolume Imgs);                                             // Converts a channel 2D images into vectors stacked together in a single matrix and puts the different channels in a volume
Volume Vects2Imgs(Matrix* Vects);                                               // Converts back a matrix of vectors into a volume of 2D images
