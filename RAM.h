#include "systemc.h"
#include "run_mode.h"
#include <iostream>
#include <fstream>


using namespace std;

SC_MODULE(RAM)
{	
	sc_in_clk clock;
	sc_in<bool> wr;
	sc_in<sc_uint<16> >ram_addr;
	sc_in<TYPE > data_in;
	sc_out<TYPE > data_out;
	TYPE RAM_[65536];
	void r_data();
	void w_data();
	
	SC_CTOR(RAM)
	{
		for(int i=0;i<65536;++i)
			RAM_[i]=0;
		SC_METHOD(r_data);
		sensitive << wr << ram_addr;
		SC_METHOD(w_data);
		sensitive << clock.neg();
	}
};


