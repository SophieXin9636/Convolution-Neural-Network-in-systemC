#include "systemc.h"
#include "run_mode.h"
#include <iostream>
#include <fstream>

using namespace std;

SC_MODULE(Monitor)
{	
	sc_in_clk clock;
	sc_in<TYPE > result;
	sc_in<bool> valid;
	int cnt_i;
	ifstream fin;

	void monitor();
	
	SC_CTOR(Monitor)
	{
		cnt_i = 0;
		fin.open(LABEL_FILE);
		cout<<"Loading label : " << LABEL_FILE << '\n';
		SC_METHOD(monitor);
		sensitive << clock.neg();
	}
};

