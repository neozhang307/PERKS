
#include <vector>
#include <string>


namespace mersenne {

/* Period parameters */
// const unsigned int N          = 624;
// const unsigned int M          = 397;
// const unsigned int MATRIX_A   = 0x9908b0df; /* constant vector a */
// const unsigned int UPPER_MASK = 0x80000000; /* most significant w-r bits */
// const unsigned int LOWER_MASK = 0x7fffffff; /* least significant r bits */

// static unsigned int mt[N];  /* the array for the state vector  */
// static int mti = N + 1;     /* mti==N+1 means mt[N] is not initialized */

/* initializes mt[N] with a seed */
void init_genrand(unsigned int s);

/* initialize by an array with array-length */
/* init_key is the array for initializing keys */
/* key_length is its length */
/* slight change for C++, 2004/2/26 */
void init_by_array(unsigned int init_key[], int key_length);


/* generates a random number on [0,0xffffffff]-interval */
unsigned int genrand_int32(void);


} // namespace mersenne


struct CommandLineArgs
{

    std::vector<std::string>    keys;
    std::vector<std::string>    values;
    std::vector<std::string>    args;
#ifdef __NVCC__
    cudaDeviceProp              deviceProp;
#endif // __NVCC__
    float                       device_giga_bandwidth;
    size_t                      device_free_physmem;
    size_t                      device_total_physmem;

    /**
     * Constructor
     */
    CommandLineArgs(int argc, char **argv) ;
        // keys(10),
        // values(10);


    /**
     * Checks whether a flag "--<flag>" is present in the commandline
     */
    bool CheckCmdLineFlag(const char* arg_name);


    /**
     * Returns number of naked (non-flag and non-key-value) commandline parameters
     */
    template <typename T>
    int NumNakedArgs();

    /**
     * Returns the commandline parameter for a given index (not including flags)
     */
    template <typename T>
    void GetCmdLineArgument(int index, T &val);


    /**
     * Returns the value specified for a given commandline parameter --<flag>=<value>
     */
    template <typename T>
    void GetCmdLineArgument(const char *arg_name, T &val);



    /**
     * Returns the values specified for a given commandline parameter --<flag>=<value>,<value>*
     */
    template <typename T>
    void GetCmdLineArguments(const char *arg_name, std::vector<T> &vals);



    /**
     * The number of pairs parsed
     */
    int ParsedArgc();


#ifdef __NVCC__

    /**
     * Initialize device
     */
    cudaError_t DeviceInit(int dev = -1);


// template <typename T>
// void DisplayDeviceResults(
//     T *d_data,
//     size_t num_items);
/**
 * Verify the contents of a device array match those
 * of a host array
 */
template <typename S, typename T>
int CompareDeviceResults(
    S *h_reference,
    T *d_data,
    size_t num_items,
    bool verbose =true,
    bool display_data =false);

/**
 * Verify the contents of a device array match those
 * of a device array
 */
template <typename T>
int CompareDeviceDeviceResults(
    T *d_reference,
    T *d_data,
    size_t num_items,
    bool verbose =true,
    bool display_data=false );

template<class TYPE>
bool areAlmostEqual(TYPE a, TYPE b, TYPE maxRelDiff) ;

#endif // __NVCC__

};
