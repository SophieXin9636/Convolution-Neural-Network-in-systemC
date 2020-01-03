#include "Lenet.h"
#include <iostream>


void lenet::lenet_proc(){
	if(reset.read()){
		step = 1, step2 = 0;
		cnt = 0; n = 0;
		cnt2 = 0;
		offset = 0;
		result = 0;
		rom_rd = true;  rom_addr = 0;
		result = 0.0;
		valid = false;
		return;
	}

	/* Input layer */
	if(step == 1){
		for(i=0; i<28; i++){
			for(j=0; j<28; j++){
				fin >> image[i][j];
				image[i][j] /= (TYPE)256.0;
			}
		}
		step++;
		rom_rd.write(true);
	}
	/* First convolution layer */
	else if(step == 2){
		if(cnt < 25){
			rom_rd.write(true);		
			kernel[cnt/5][cnt%5] = rom_data_in.read();
			cnt++;
			rom_addr.write(26 * step2 + cnt);
		}
		else if(cnt == 25){
			rom_rd.write(true);
			bias = rom_data_in.read();
			cnt++;
			rom_addr.write(26 * step2 + cnt);
			/*
			cout << "kernel " << step2 + 1 << " :" << endl;
			for(i=0; i<5; i++){
				for(j=0; j<5; j++){
					cout << kernel[i][j] <<" ";
				}
				cout << endl;
			}
			cout << "Bias: " << bias << endl << endl;*/
		}
		else{
			if(1){
				sum[step2][n] = 0.0;
				/* convolution */
				for(i=n/24, ka=0; i<n/24+5, ka<5; i++, ka++){
					for(j=n%24, kb=0; j<n%24+5, kb<5; j++, kb++){
						sum[step2][n] += image[i][j] * kernel[ka][kb];
					}
				}
				//cout << endl << endl;
				// add bias
				sum[step2][n] += bias;

				/* activation function: ReLu Function */
				if(sum[step2][n] <= 0){
					sum[step2][n] = 0;
				}
				
				// write into RAM
				ram_wr.write(0);
				ram_addr.write(step2 * 24*24 + n);
				ram_data_out.write(sum[step2][n]);

				//cout <<"Lanet: " << ram_addr <<" " << ram_data_out << endl;
				//cout <<"  n  : " << n <<" "<< sum[step2][n] << endl;

				n++;

				if(n % (24*24) == 0){
					cnt = 0;
					step2++;
					n = 0;
					if(step2 == 6) // sixth time
						step++;
					return;
				}
			}
		}
	}
	else if(step == 3){
		
	}
}