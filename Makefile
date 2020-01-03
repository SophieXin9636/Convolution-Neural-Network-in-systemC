LIB_DIR=/usr/systemc/lib-linux64
INC_DIR=/usr/systemc/include
LIB=-lsystemc -lm -DSC_INCLUDE_FX
#RPATH=-Wl,-rpath=/usr/local/systemc-2.3.1a/lib-linux64

O = test
C = *.cpp

all:
	g++ -I . -I  $(INC_DIR) -L . -L $(LIB_DIR) -o $(O) $(C) $(LIB)

clean:
	rm -rf $(O)
