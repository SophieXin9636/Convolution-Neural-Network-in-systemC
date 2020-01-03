#include "Lenet.h"
#include "ROM.h"
#include "RAM.h"
#include "Monitor.h"
#include "clockreset.h"
#include <iostream>

using namespace std;

int sc_main(int argc,char* argv[])
{
	cout<<"\n\n\n";
	sc_signal<bool> clk,reset;
	sc_signal<bool> rom_rd;
	sc_signal<bool> ram_wr;
	sc_signal<sc_uint<16> > rom_addr;
	sc_signal<sc_uint<16> > ram_addr;
	sc_signal<TYPE > rom_data;
	sc_signal<TYPE > ram_data_in;
	sc_signal<TYPE > ram_data_out;
	sc_signal<TYPE > result;
	sc_signal<bool> output_valid;
	
	
	ROM rom("Rom");
	RAM ram("Ram");
	lenet net("lenet");
	Monitor mon("mon");
	Clock clock("clock",5,10*1000*1000);
	Reset rst("reset",20);
	
	clock(clk);
	rst(reset);
	rom(clk,rom_rd,rom_addr,rom_data);
	ram(clk,ram_wr,ram_addr,ram_data_out,ram_data_in);
	net(clk,reset,rom_rd,ram_wr,rom_addr,ram_addr,rom_data,ram_data_in,ram_data_out,result,output_valid);
	mon(clk,result,output_valid);
	cout << '\n';
	
	sc_start(100*1000*1000,SC_NS);
	if(mon.cnt_i != 10)
		cout << "Reaching time limit.\n";
	else
		cout << "10 output recived.\n";
	
	cout<<"\n\n\n";
	return 0;
}











