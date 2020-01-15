#include "systemc.h"
#include "run_mode.h"
#include<iostream>
#include<fstream>

#define KERNEL_SIZE 5
#define DEPTH_L1 6
#define DEPTH_L2 16

using namespace std;

SC_MODULE(lenet)
{	
	sc_in_clk clock;
	sc_in<bool> reset;
	sc_out<bool> rom_rd;
	sc_out<bool> ram_wr;
	sc_out<sc_uint<16> > rom_addr;
	sc_out<sc_uint<16> > ram_addr;
	sc_in<TYPE > rom_data_in;
	sc_in<TYPE > ram_data_in;
	sc_out<TYPE > ram_data_out;
	sc_out<TYPE > result;
	sc_out<bool> valid;
	

	void lenet_proc();
	
	ifstream fin;
	int ram_w_cur; // RAM write of current index
	int ram_r_cur; // RAM read  of current index
	int rom_cur;
	TYPE image[28][28];
	TYPE kernel[KERNEL_SIZE][KERNEL_SIZE];
	TYPE bias, sum;
	int i, j, ka, kb, cnt, cnt2, n, step;
	int times;

	TYPE scopeMAX[4]; // MAX pool
	int pool_idx;

	TYPE kernel_L2[6][KERNEL_SIZE][KERNEL_SIZE];
	TYPE pooling_ft_L1[6][12][12]; // First pooling feature

	TYPE input[4*4*16]; // input layer
	TYPE weight[256];
	
	SC_CTOR(lenet)
	{
		fin.open(INPUT_FILE);


		SC_METHOD(lenet_proc);
		sensitive << clock.pos();
	}
};

