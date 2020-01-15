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
			/* read kernel data from ROM 0~155 */
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
				ram_addr.write(ram_r_cur++); // start from 0
				i = 0;
				//cout <<"Step 2 has done\n";
				//cout << "index: " << rom_cur << endl;
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
				/* write into RAM 0~3455 */
				ram_wr.write(0); // write
				//cout << "RAM: " << ram_w_cur <<"\t" << sum << endl;
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
			ram_r_cur = 3457;
			cnt = 0; n = 0;
			times = 0;
			ram_wr.write(1); // read data
			ram_addr.write(ram_r_cur);
			rom_rd.write(true); // read Second convolution layer data from ROM
			//cout <<"Step 3 has done\n";
			//cout << "index: " << rom_cur << endl;
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
				//cout << ram_w_cur-3456 <<" "<<  max << endl;
				// store (write) into RAM 3456~4319
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
				ram_wr.write(1); // read 0~3455
				ram_addr.write(pool_idx);
			}
		}
	}
	/* Second convolution layer: 8*8*16 times */
	else if(step == 4){
		if(times == 16){ // 16th time
			step++;
			cnt = 0;
			times = 0;
			ram_r_cur = 4320;
			pool_idx = ram_r_cur;
			ram_wr.write(1); // read
			ram_addr.write(ram_r_cur++);
			i = 0;
			//cout <<"Step 4 has done\n";
			//cout << "index: " << rom_cur << endl;
		}
		else {
			if(cnt < 864){ // 12*12*6
				/* read pooling data from RAM */ // 3456~4319
				pooling_ft_L1[cnt/144][(cnt%144)/12][cnt%12] = ram_data_in.read();
				//cout <<"RAM: " << ram_r_cur <<"\t"<<cnt <<"\t" << i <<"\t"<<pooling_ft_L1[cnt/144][(cnt%144)/12][cnt%12] << endl;
				ram_wr.write(1); // read
				ram_addr.write(ram_r_cur++);
				cnt++;
			}
			else if(cnt < 1014){ // 12*12*6 + 5*5*6
				int t = cnt-864;
				/* read kernel data from ROM 156~2571 */
				rom_rd.write(true);
				kernel_L2[t/25][(t%25)/5][t%5] = rom_data_in.read();
				//cout <<"ROM: " << rom_cur <<"\t"<< "kernel[" << t/25 <<"][" << (t%25)/5 <<"][" << t%5  <<"]="<< kernel_L2[t/25][(t%25)/5][t%5] << endl;
				rom_addr.write(rom_cur++);
				cnt++;
			}
			else if(cnt == 1014){ // 12*12 + 151
				/* read bias data from ROM */
				rom_rd.write(true);
				bias = rom_data_in.read();
				//cout <<"ROM: " << rom_cur <<"\t" << times <<"\t"<< bias << endl; 
				rom_addr.write(rom_cur++);
				cnt++;
			}
			else{
				//if(n == 0) cout <<"Layer: " << times << endl;
				/* convolution */
				sum = (TYPE)0.0;
				for(int k=0; k<DEPTH_L1; k++){
					for(i=n/8, ka=0; i<n/8+5, ka<5; i++, ka++){
						for(j=n%8, kb=0; j<n%8+5, kb<5; j++, kb++){
							sum += pooling_ft_L1[k][i][j] * kernel_L2[k][ka][kb];
							//cout << pooling_ft_L1[k][i][j]<< endl;
						}
					}
				}
				// add bias
				sum += bias;

				/* activation function: ReLu Function */
				if(sum <= (TYPE)0.0){
					sum = (TYPE)0.0;
				}

				/* write into RAM 4320~5343 */
				ram_wr.write(0); // write
				ram_addr.write(ram_w_cur++);
				ram_data_out.write(sum);
				//cout << sum << " ";
				n++;
				//if(n % 8 == 0) cout << endl;

				if(n % 64 == 0){
					times++;
					cnt = 864;
					n = 0;
					//cout << endl;
				}
			}
		}
	}
	/* Second Pooling Layer (Subsampling): Max pool 4*4*16 times */
	else if(step == 5){
		int next_pool_dir[4] = {1,8,9,2};
		if(times == 16){
			step++;
			cnt = 0; n = 0;
			times = 0;
			ram_wr.write(1); // read data
			ram_r_cur = 5344;
			ram_addr.write(ram_r_cur++);
			rom_rd.write(true); // read layer data from ROM
			//cout <<"Step 5 has done\n";
			//cout << "index: " << rom_cur << endl;
		}
		else{
			if(cnt / 4 == 1){
				// find Max 
				TYPE max = (TYPE)-1000.0;
				for(int k=0; k<4; k++){
					//cout << k <<"\t" << scopeMAX[k] << endl;
					if(scopeMAX[k] > max){
						max = scopeMAX[k];
					}
				}
				//cout << ram_w_cur-5344 <<" "<< max << endl;
				// store (write) into RAM 5344~5599
				ram_wr.write(0); // write
				ram_addr.write(ram_w_cur++);
				ram_data_out.write(max);

				cnt = 0;
				i += 2;
				if(i % 8 == 0){
					i += 8;
					if(i % 64 == 0){
						times++;
					}
				}
			}
			else{
				/* read from RAM 4320~5343 */
				scopeMAX[cnt] = ram_data_in.read();
				//cout << cnt <<" "<< pool_idx-4320 <<" "<< scopeMAX[cnt] << endl;
				if(cnt==3 && i%8==6) pool_idx = 4320 + i + 8 + next_pool_dir[cnt];
				else pool_idx = 4320 + i + next_pool_dir[cnt];
				cnt++;
				ram_wr.write(1); // read
				ram_addr.write(pool_idx);
			}
		}
	}
	/* First fully connected (Flatten) layer: 128 times */
	else if(step == 6){
		if(times == 128){ // 128th times
			step++;
			cnt = 0;
			times = 0;
			ram_wr.write(1); // read
			ram_r_cur = 5600;
			ram_addr.write(ram_r_cur++);
			rom_rd.write(true); // read layer data from ROM
			//cout <<"Step 6 has done\n";
			//cout << "index: " << rom_cur << endl;
		}
		else{
			if(cnt < 256){ // read from 5344~5599 (256) Second pooling data
				/* read from RAM */
				input[cnt] = ram_data_in.read();
				ram_wr.write(1); // read
				//cout << "RAM "<< ram_r_cur << "\t " << cnt <<"\t" << input[cnt] << endl; 
				ram_addr.write(ram_r_cur++);
				cnt++;
			}
			else if(cnt < 512){ // 16*16
				/* read flatten data from ROM */
				rom_rd.write(true);		
				weight[cnt-256] = rom_data_in.read();
				rom_addr.write(rom_cur++);
				cnt++;
			}
			else if(cnt == 512){
				/* read bias data from ROM */
				rom_rd.write(true);
				bias = rom_data_in.read();
				rom_addr.write(rom_cur++);
				cnt++;
			}
			else if(cnt == 513){
				sum = (TYPE)0.0;
				/* flatten */
				for(i=0; i<256; i++){
					sum += input[i] * weight[i];
				}
				// add bias
				sum += bias;

				/* activation function: ReLu Function */
				if(sum <= (TYPE)0.0){
					sum = (TYPE)0.0;
				}
				//cout << times <<" " << sum << endl;	
				/* write into RAM */
				ram_wr.write(0); // write into 5600~5727 (128)
				ram_addr.write(ram_w_cur++);
				ram_data_out.write(sum);
				times++;
				cnt = 256;
			}
		}
	}
	/* Second fully connected (Flatten) layer: 84 times */
	else if(step == 7){
		if(times == 84){ // 84th times
			step++;
			cnt = 0;
			times = 0;
			ram_r_cur = 5728;
			ram_wr.write(1); // read
			ram_addr.write(ram_r_cur++);
			rom_rd.write(true); // read layer data from ROM
			//cout <<"Step 7 has done\n";
			//cout << "index: " << rom_cur << endl;
		}
		else{
			if(cnt < 128){ // read from 5600~5727 (128) Second pooling data
				/* read from RAM */
				input[cnt] = ram_data_in.read();
				//cout << "RAM "<< ram_r_cur << "\t " << cnt <<"\t" << input[cnt] << endl; 
				ram_wr.write(1); // read
				ram_addr.write(ram_r_cur++);
				cnt++;
			}
			else if(cnt < 256){ // 128
				/* read flatten data from ROM */
				rom_rd.write(true);		
				weight[cnt-128] = rom_data_in.read();
				rom_addr.write(rom_cur++);
				cnt++;
			}
			else if(cnt == 256){
				/* read bias data from ROM */
				rom_rd.write(true);
				bias = rom_data_in.read();
				rom_addr.write(rom_cur++);
				cnt++;
			}
			else if(cnt == 257){
				sum = (TYPE)0.0;
				/* flatten */
				for(i=0; i<128; i++){
					sum += input[i] * weight[i];
				}
				// add bias
				sum += bias;

				/* activation function: ReLu Function */
				if(sum <= (TYPE)0.0){
					sum = (TYPE)0.0;
				}
				//cout << sum <<"   ";
				//if(times % 7 == 6) cout << endl;		
				/* write into RAM */
				ram_wr.write(0); // write 5728~5811 (84)
				ram_addr.write(ram_w_cur++);
				ram_data_out.write(sum);
				times++;
				cnt = 128;
			}
		}
	}
	/* Third fully connected (Flatten) of Output layer: 10 times */
	else if(step == 8){
		if(times == 10){ // 10 times
			step++;
			//cout <<"Step 8 has done\n";
			//cout << "index: " << rom_cur << endl;
		}
		else{
			if(cnt < 84){ // read from 5727~5811 (84) Second pooling data
				/* read from RAM */
				input[cnt] = ram_data_in.read();
				//cout << "RAM "<< ram_r_cur << "\t " << cnt <<"\t" << input[cnt] << endl; 
				ram_wr.write(1); // read
				ram_addr.write(ram_r_cur++);
				cnt++;
			}
			else if(cnt < 168){ // 84
				valid.write(false);
				/* read flatten data from ROM */
				rom_rd.write(true);		
				weight[cnt-84] = rom_data_in.read();
				//rom_addr.write(rom_cur++);
				if(rom_cur<47154)
					rom_addr.write(rom_cur++);
				cnt++;
			}
			else if(cnt == 168){
				/* read bias data from ROM */
				rom_rd.write(true);
				bias = rom_data_in.read();
				//rom_addr.write(rom_cur++);
				if(rom_cur<47154)
					rom_addr.write(rom_cur++);
				cnt++;
			}
			else if(cnt == 169){
				sum = (TYPE)0.0;
				/* flatten */
				for(i=0; i<84; i++){
					sum += input[i] * weight[i];
					//cout << "i:  "<< i << "\t " << input[i] <<"\t\t" << weight[i] <<" = "  <<sum<< endl;
				}
				// add bias
				sum += bias;

				valid.write(true);
				result.write(sum);
				times++;
				cnt = 84;
			}
		}
	}
}
