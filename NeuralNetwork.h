#pragma once
#ifndef CNN_HEADER
#define CNN_HEADER
#include "Dictionary.h"
#include "Matrix.h"
#include "Volume.h"
#include "VectVolume.h"
#include "NN_Tools.h"
#include "Activations.h"
#include "ConvFeedForward.h"
#include "ConvBackProb.h"
#include <thread>

typedef Dictionary<string, Matrix*> Mat_Dictionary;
typedef Dictionary<string, Volume> Vol_Dictionary;
typedef Dictionary<string, VectVolume > VectVol_Dictionary;

class NeuralNetwork
{
private:
	Arguments* Arg;
	bool momentum;				  // Indicates whether momentum is used or not
	bool isLastepoch;			  // Label for the last epoch
	int  Cores;                   // The number of allowed threads in the current hardware for maximum efficiency
    Matrix*** D;		          // Dropout Matrices in fully connected layers
	matrix<bool>**** D2;          // DropConnect Matrices in fully connected layers

	//Convolution Dictionaries
	VectVol_Dictionary  Conv_Weights;         // Dictionary containing weights of convolution layers
	VectVol_Dictionary* Conv_Cache;	          // Dictionary containing temporaral values of internal activations of convolution layers
	Mat_Dictionary*     Conv_Cache_Mat;          // Dictionary containing mean & var for Conv layers

	VectVol_Dictionary* Conv_Grades;          // Dictionary containing gradients of weights and biases of convolution layers
	VectVol_Dictionary  ADAM_dWC;		      // Dictionary containing ADAM dW gradients of conv layers
	Mat_Dictionary      Conv_biases;          // Dictionary containing biases of convolution layers
	Mat_Dictionary*     Conv_dbiases;         // Dictionary containing biases of convolution layers
	Mat_Dictionary      ADAM_dbC;		      // Dictionary containing ADAM db gradients of conv layers
	//Fully connected Dictionaries
	Mat_Dictionary      FC_Parameters;		  // Dictionary containing weights and biases of fully connected layers
	Mat_Dictionary*     FC_Cache;		      // Dictionary containing temporaral values of internal activations of fully connected layers
	Mat_Dictionary*     FC_Grades;			  // Dictionary containing gradients of weights and biases of fully connected layers
	Mat_Dictionary      FC_ADAM;              // Dictionary containing ADAM gradients

public:
    // Interface functions
	NeuralNetwork(Arguments* A);              // Initialize the network with arguments A
	void train();                             // Begin training
	void continueTrainFC();                   // Continue training in infinte loop (For debug only!)
	void test(Mode devOrtest);                // Test the network

private:
    // Fully connected main functions
    void init_FC();                           // Initialize the fully connected network
    void train_FC();                          // Train the fully connected network in a single thread
    void train_FC_thread();                   // Train the fully connected network in multi threads
	Matrix* test_FC(Mode mode);               // Test the fully connected network on either dev or test sets

    // Convolution LENET1 main functions
	void init_LeNet1();                       // Initialize the convolution LENET1 network
	void train_LeNet1();                      // Train the convoltion LENET1 network
	void train_LeNet1_thread();               // Train the convoltion LENET1 network in multi threads
	Matrix* test_LeNet1(Mode mode);           // Test the convoltion LENET1 network on either dev or test sets

	// Convolution LENET1 main functions
	void init_other();                         // Initialize the convolution other network
	void train_other();                        // Train the convoltion other network
	void train_other_thread();                 // Train the convoltion other network in multi threads
	Matrix* test_other(Mode mode);             // Test the convoltion other network on either dev or test sets

    // Fully connected feed forward and back propagation functions
	Matrix* FC_FeedForward(Mode mode, int ThreadNum);
	void    FC_CalGrads(Matrix* cur_Y, Matrix* Y_hat, int ThreadNum);
	void    FC_UpdateParameters(int iteration, int ThreadNum);

	// Convolution feed forward and back propagation functions
	void convLayer(Mode mode, int stride, int A_index, ActivationType activation, int ThreadNum);
	void poolLayer(int stride,int f, Mode mode,int A_index, int ThreadNum);
	void ConvBackward(int stride, int A_index, ActivationType activation, int ThreadNum);
	void ConvBackwardOptimized(int stride,int A_index, ActivationType activation, int ThreadNum);
	void pool_backward(int f,int stride, Mode mode,int A_index, int ThreadNum);
    void Conv_updateparameters (int iteration, int W_index, int ThreadNum);
	void Conv_BN_feedforward(Mode mode,int index,int ThreadNum);
	void Conv_BN_backprop(VectVolume dZC, int index, int ThreadNum);

	// Averaging the grades in multi-threadded operation
	void Average();

private:
    // The function executing the thread in BatchMultiThread
	class Thread_Initialize
	{
	public:
		void operator()(NeuralNetwork* NN, int ThreadNum, int start, int end);
	};
	friend class Thread_Initialize;     // To access the private members of NeuralNetwork
};
#endif // !CNN_HEADER

