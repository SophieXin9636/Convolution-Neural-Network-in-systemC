#include "systemc.h"
#include "run_mode.h"
#include <iostream>
#include <fstream>


using namespace std;

SC_MODULE(ROM)
{	
	sc_in_clk clock;
	sc_in<bool> ird;
	sc_in<sc_uint<16> >rom_addr;
	sc_out<TYPE > data;
	TYPE ROM_[47154];
	void read_data();
	
	SC_CTOR(ROM)
	{
		float buf;
		int cnt = 0;
		ifstream win;
		
		cout << "Loading weights...\n";
		win.open("lenet_weights/lenet_weight_c1.txt");
		win >> buf;
		while(!win.eof())
		{
			ROM_[cnt++] = buf;
			win >> buf;
		}
		win.close();
		
		win.open("lenet_weights/lenet_weight_c2.txt");
		win >> buf;
		while(!win.eof())
		{
			ROM_[cnt++] = buf;
			win >> buf;
		}
		win.close();
		
		win.open("lenet_weights/lenet_weight_d1.txt");
		win >> buf;
		while(!win.eof())
		{
			ROM_[cnt++] = buf;
			win >> buf;
		}
		win.close();
		
		win.open("lenet_weights/lenet_weight_d2.txt");
		win >> buf;
		while(!win.eof())
		{
			ROM_[cnt++] = buf;
			win >> buf;
		}
		win.close();
		
		win.open("lenet_weights/lenet_weight_d3.txt");
		win >> buf;
		while(!win.eof())
		{
			ROM_[cnt++] = buf;
			win >> buf;
		}
		win.close();
		
		cout << cnt << " loaded weights.\n";
		
		SC_METHOD(read_data);
		sensitive << ird << rom_addr;
	}
};


