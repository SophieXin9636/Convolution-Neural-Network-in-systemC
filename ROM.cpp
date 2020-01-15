#include "ROM.h"


void ROM::read_data(){
	
	if(ird.read()){
		data = ROM_[rom_addr.read()];
		int n = rom_addr.read();
		//if(n > 46304)
		//cout << "ROM " << n <<": " << data << endl;
	}
}



