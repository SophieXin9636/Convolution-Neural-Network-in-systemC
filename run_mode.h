#define MODE 1
//1 for floating , 2 for fixed

#define NUM 1







#define QUO(x) #x
#define QUOTE(x) QUO(x)
#define CONC(x,y,z) x ## y.z
#define CONCAT(x,y) QUOTE(CONC(x,y,txt))

#if MODE == 1
	#define TYPE float
#elif MODE == 2
	#define TYPE sc_fixed<17,9,SC_RND,SC_SAT>
#endif

#define INPUT_FILE CONCAT(Pattern/input,NUM)
#define LABEL_FILE CONCAT(Answer/lenet_output_,NUM)
