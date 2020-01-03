#include "Monitor.h"
#include <iomanip>


void Monitor:: monitor(){
	if(valid)
	{
		float label;
		fin >> label;
		cout.setf(ios::fixed);
		float ans = result.read();
		cout << "Output #" << setw(3) << cnt_i << " : " << setw(12) << ans << "\tLabel : " << setw(12) << label << '\n';
		cout.unsetf(ios::fixed);
		cnt_i = cnt_i + 1;
		if(cnt_i == 10)
			sc_stop();
	}
	
}


