#include "Lenet.h"
#include <iostream>


void lenet::lenet_proc(){
	if(reset.read()){
		step = 1, times = 0;
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
	/* First convolution layer: 24*24*6 times */
	else if(step == 2){
		if(cnt < 25){
			/* read kernel data from ROM */
			rom_rd.write(true);		
			kernel[cnt/5][cnt%5] = rom_data_in.read();
			cnt++;
			rom_addr.write(26 * times + cnt);
		}
		else if(cnt == 25){
			/* read bias data from ROM */
			rom_rd.write(true);
			bias = rom_data_in.read();
			cnt++;
			rom_addr.write(26 * times + cnt);
			/*
			cout << "kernel " << times + 1 << " :" << endl;
			for(i=0; i<5; i++){
				for(j=0; j<5; j++){
					cout << kernel[i][j] <<" ";
				}
				cout << endl;
			}
			cout << "Bias: " << bias << endl << endl;*/
		}
		else{
			sum[times][n] = 0.0;
			/* convolution */
			for(i=n/24, ka=0; i<n/24+5, ka<5; i++, ka++){
				for(j=n%24, kb=0; j<n%24+5, kb<5; j++, kb++){
					sum[times][n] += image[i][j] * kernel[ka][kb];
				}
			}
			//cout << endl << endl;
			// add bias
			sum[times][n] += bias;

			/* activation function: ReLu Function */
			if(sum[times][n] <= 0){
				sum[times][n] = 0;
			}
				
			/* write into RAM */
			ram_wr.write(0); // write
			ram_addr.write(times * 24*24 + n);
			ram_data_out.write(sum[times][n]);
			n++;

			if(n % (24*24) == 0){
				cnt = 0;
				times++;
				n = 0;
				if(times == 6){ // sixth time
					step++;
					cnt = 0;
					ram_wr.write(1); // read
					ram_addr.write(0);
					i = 0;
					ram_cur = times * 24*24 + n;
					times = 0;
				}
				return;
			}
		}
	}
	/* Pooling Layer: Max pool 12*12*6 times */
	else if(step == 3){
		int next_pool_dir[4] = {1,24,25,2};
		if(cnt / 4 == 1){
			// find Max 
			TYPE max = -1000.0;
			for(int k; k<4; k++){
				if(scopeMAX[k] > max){
					max = scopeMAX[k];
				}
			}
			// store into RAM
			ram_wr.write(0); // write
			ram_addr.write(ram_cur++);
			ram_data_out.write(max);

			cnt = 0;
			i += 2;
			if(i % 24 == 0){
				i += 24;
				if(i % 576 == 0){
					times++;
				}
			}
		}
		else{
			/* read from RAM */
			scopeMAX[cnt] = ram_data_in.read();
			pool_idx = i + next_pool_dir[cnt];
			cnt++;
			ram_wr.write(1); // read
			ram_addr.write(pool_idx++);
		}
		if(times == 6){
			step++;
		}
	}
	/* Second convolution layer: 8*8*16 times */
	else if(step == 4){
		
	}
}