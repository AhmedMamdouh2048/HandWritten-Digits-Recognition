#ifndef NN_TOOLS_H_INCLUDED
#define NN_TOOLS_H_INCLUDED
#include <iostream>
#include "Matrix.h"
#include "Activations.h"
#include "Volume.h"
#include "ConvFeedForward.h"
#include <thread>
typedef matrix<float> Matrix;
typedef matrix<uint16_t> IntMatrix;
////////////////////////////////////////////////////////
enum TypeOfNet {FC,LENET1,other};
enum Optimizer {ADAM,GRADIENT_DESCENT};
enum ErrorType {SQAURE_ERROR,CROSS_ENTROPY};
enum Mode {TRAIN,DEV,TEST,MAX,AVG};
////////////////////////////////////////////////////////
struct layer
{
    float neurons;
    ActivationType activation;
    float numOfLinearNodes;

    void put(float n, ActivationType activ)
    {
        neurons=n;
        activation=activ;
    }
};
////////////////////////////////////////////////////////
struct Arguments
{
    TypeOfNet NetType;                  // Type of neural network. Either FC (fully connected) or LENET1 (convolutional)
    layer* layers;                      // The activations and number of neurons in each layer
    int numOfLayers;                    // The number of layers
    IntMatrix* X;                       // The input dataset
    Matrix* Y;                          // The labels of the input dataset
    Matrix* X_dev;                      // The development set
    Matrix* Y_dev;                      // The labels of the development dataset
    Matrix* X_test;                     // The test set
    Matrix* Y_test;                     // The labels of the test set
    float learingRate;                  // The learning rate alpha
    float decayRate;                    // The decay of learining rate
    int numOfEpochs;                    // The total number of epochs required for training
    int batchSize;                      // The batch size
    Optimizer optimizer;                // The type of optimizer. Either Gradient Descent or ADAM
    int numPrint;                       // The rate of printing the accuracy on the screen
    ErrorType ErrType;                  // The type of error used. Either square error or cross entropy
    float regularizationParameter;      // The regularization parameter lambda
    bool batchNorm;                     // Is batch normalization activated or not
    bool dropout;                       // Is drop-out activated or not
    bool dropConnect;                   // Is drop-connection activated or not
    float* keep_prob;                   // The probabilty distribution of drop-out and drop-connection across the layers
    bool BatchMultiThread;              // Is multi-threadding enabled or not
    bool EB;                            // Is efficient backprop enabled or not
    string path;                        // The path of the file at which the parameters are saved at
};
////////////////////////////////////////////////////////
std::string CharGen(std::string name, int i);
////////////////////////////////////////////////////////
void AccuracyTest(Matrix* Y, Matrix* Y_hat, string devOrtest);
void AccuracyTest(Matrix* Y, Matrix* Y_hat, Matrix* errors);
////////////////////////////////////////////////////////
Matrix* DOT (Matrix* X, Matrix* Y);
void DotPart(int part, Matrix* result, Matrix* X, Matrix* Y);
////////////////////////////////////////////////////////
void unlearned_patterns(int& ncols, Matrix* Y, Matrix* Y_hat, int*indices, int minibatch_num, int minibatch_size, float StudiedWell = 1);
Matrix* reduced_X(Matrix*X,int* indices,int ncol);
Matrix* reduced_Y(Matrix*Y,int* indices,int ncols);
void getIndices(Matrix* Y,int pattern, uint32_t* indices, int& numOfPatterns);
void getPattern(Volume& X_pattern, Matrix*& Y_pattern, Matrix* X, Matrix* Y, int pattern);
void MIX(Matrix*& X, Matrix*& Y, Matrix* X_, Matrix* Y_);
////////////////////////////////////////////////////////
#endif // NN_TOOLS_H_INCLUDED
