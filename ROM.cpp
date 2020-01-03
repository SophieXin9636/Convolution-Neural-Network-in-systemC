#include "ROM.h"


void ROM::read_data(){
	
	if(ird.read())
		data = ROM_[rom_addr.read()];


}



