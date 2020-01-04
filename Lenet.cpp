#include "Lenet.h"
#include <iostream>


void lenet::lenet_proc(){
	if(reset.read()){
		step = 1, times = 0;
		cnt = 0; n = 0;
		cnt2 = 0;
		rom_cur = 0;
		ram_r_cur = 0;
		ram_w_cur = 0;
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
		rom_addr.write(rom_cur++);
	}
	/* First convolution layer: 24*24*6 times */
	else if(step == 2){
		if(cnt < 25){
			/* read kernel data from ROM */
			rom_rd.write(true);		
			kernel[cnt/5][cnt%5] = rom_data_in.read();
			cnt++;
			rom_addr.write(rom_cur++);
		}
		else if(cnt == 25){
			/* read bias data from ROM */
			rom_rd.write(true);
			bias = rom_data_in.read();
			cnt++;
			rom_addr.write(rom_cur++);
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
			if(times == 6){ // sixth time
				step++;
				cnt = 0;
				times = 0;
				ram_wr.write(1); // read
				ram_addr.write(ram_r_cur); // start from 0
				i = 0;
			}
			else{
				sum = 0.0;
				/* convolution */
				for(i=n/24, ka=0; i<n/24+5, ka<5; i++, ka++){
					for(j=n%24, kb=0; j<n%24+5, kb<5; j++, kb++){
						sum += image[i][j] * kernel[ka][kb];
					}
				}
				// add bias
				sum += bias;

				/* activation function: ReLu Function */
				if(sum <= 0){
					sum = 0;
				}
					
				n++;
				/* write into RAM */
				ram_wr.write(0); // write
				ram_addr.write(ram_w_cur++);
				ram_data_out.write(sum);
				if(n % (24*24) == 0){
					times++;
					cnt = (times < 6)? 0: cnt;
					n = 0;
				}
			}
		}
	}
	/* Pooling Layer (Subsampling): Max pool 12*12*6 times */
	else if(step == 3){
		int next_pool_dir[4] = {1,24,25,2};
		if(times == 6){
			step++;
			ram_r_cur = pool_idx;
			cnt = 0; n = 0;
			times = 0;
			ram_wr.write(1); // read data
			ram_addr.write(ram_r_cur);
			rom_rd.write(true); // read Second convolution layer data from ROM
		}
		else{
			if(cnt / 4 == 1){
				// find Max 
				TYPE max = (TYPE)-1000.0;
				for(int k=0; k<4; k++){
					if(scopeMAX[k] > max){
						max = scopeMAX[k];
					}
				}
				// store (write) into RAM
				ram_wr.write(0); // write
				ram_addr.write(ram_w_cur++);
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
		}
	}
	/* Second convolution layer: 8*8*16 times */
	else if(step == 4){
		if(cnt < 144){ // 12*12
			/* read pooling data from RAM */
			pooling_ft_L1[cnt/12][cnt%12] = ram_data_in.read();
			ram_wr.write(1); // read
			ram_addr.write(ram_r_cur++);
			cnt++;
		}
		else if(cnt < 294){ // 12*12 + 5*5*6
			int t = cnt-144;
			/* read kernel data from ROM */
			rom_rd.write(true);		
			kernel_L2[t/25][(t%25)/5][t%5] = rom_data_in.read();
			rom_addr.write(rom_cur++);
			cnt++;
		}
		else if(cnt == 294){ // 12*12 + 151
			/* read bias data from ROM */
			rom_rd.write(true);
			bias = rom_data_in.read();
			rom_addr.write(rom_cur++);
			cnt++;
			/*
			cout << "kernel " << times + 1 << " :" << endl;
			for(i=0; i<5; i++){
				for(j=0; j<5; j++){
					cout << kernel_L2[i][j] <<" ";
				}
				cout << endl;
			}
			cout << "Bias: " << bias << endl << endl;*/
		}
		else{
			if(times == 16){ // 16th time
				step++;
				cnt = 0;
				times = 0;
				ram_wr.write(1); // read
				ram_addr.write(4320);
				//ram_addr.write(ram_r_cur);
				i = 4320;
			}
			else{
				/* convolution */
				sum = (TYPE)0.0;
				for(i=n/8, ka=0; i<n/8+5, ka<5; i++, ka++){
					for(j=n%8, kb=0; j<n%8+5, kb<5; j++, kb++){
						for(int k=0; k<DEPTH_L1; k++){
							sum += pooling_ft_L1[i][j] * kernel_L2[k][ka][kb];
						}
					}
				}
				// add bias
				sum += bias;

				/* activation function: ReLu Function */
				if(sum <= 0.0){
					sum = 0.0;
				}

				/* write into RAM */
				ram_wr.write(0); // write
				ram_addr.write(ram_w_cur++);
				ram_data_out.write(sum);
				n++;

				if(n % (8*8) == 0){
					times++;
					cnt = (times < 16)? 144: cnt;
					n = 0;
				}
			}
		}
	}
	/* Second Pooling Layer (Subsampling): Max pool 4*4*16 times */
	else if(step == 5){
		int next_pool_dir[4] = {1,9,10,2};
		if(times == 16){
			step++;
			ram_r_cur = pool_idx;
			cnt = 0; n = 0;
			times = 0;
			ram_wr.write(1); // read data
			ram_addr.write(ram_r_cur++);
			rom_rd.write(true); // read layer data from ROM
		}
		else{
			if(cnt / 4 == 1){
				// find Max 
				TYPE max = (TYPE)-1000.0;
				for(int k=0; k<4; k++){
					if(scopeMAX[k] > max){
						max = scopeMAX[k];
					}
				}
				// store (write) into RAM
				ram_wr.write(0); // write
				ram_addr.write(ram_w_cur++);
				ram_data_out.write(max);

				cnt = 0;
				i += 2;
				if((i-4320) % 8 == 0){
					i += 8;
					if((i-4320) % 64 == 0){
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
				ram_addr.write(pool_idx);
			}
		}
	}
	/* First fully connected layer:  times */
	else if(step == 6){

	}
}