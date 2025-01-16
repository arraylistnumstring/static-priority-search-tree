#ifndef GPU_TREE_NODE_H
#define GPU_TREE_NODE_H

class GPUTreeNode
{
	public:
		// Current index is this-root
		// Addition and subtraction (+, -) have higher precedence than bitshift operators (<<, >>)
		// NVIDIA compiler does not inline functions in separate compilation units, so use __forceinline__ to guarantee that standard local memory-consuming function calls and returns are not used
		__forceinline__ __host__ __device__ static size_t getLeftChild(const size_t &index) {return (index << 1) + 1;};
		__forceinline__ __host__ __device__ static size_t getRightChild(const size_t &index) {return (index << 1) + 2;};
		__forceinline__ __host__ __device__ static size_t getParent(const size_t &index) {return index - 1 >> 1;};
		// From the specification of C, pointers are const if the const qualifier appears to the right of the corresponding *
		__forceinline__ __host__ __device__ static bool hasChildren(const unsigned char bitcode)
		{
			// Bitwise and (&) has higher precedence than bitwise or (|)
			return static_cast<bool> (bitcode & (HAS_LEFT_CHILD | HAS_RIGHT_CHILD));
		};
		__forceinline__ __host__ __device__ static bool hasLeftChild(const unsigned char bitcode)
			{return static_cast<bool> (bitcode & HAS_LEFT_CHILD);};
		__forceinline__ __host__ __device__ static bool hasRightChild(const unsigned char bitcode)
			{return static_cast<bool> (bitcode & HAS_RIGHT_CHILD);};
		__forceinline__ __device__ static void setLeftChild(unsigned char *const bitcodes_root, const size_t &index)
			{bitcodes_root[index] |= HAS_LEFT_CHILD;};
		__forceinline__ __device__ static void setRightChild(unsigned char *const bitcodes_root, const size_t &index)
			{bitcodes_root[index] |= HAS_RIGHT_CHILD;};
		__forceinline__ __device__ static void unsetLeftChild(unsigned char *const bitcodes_root, const size_t &index)
			{bitcodes_root[index] &= ~HAS_LEFT_CHILD;};
		__forceinline__ __device__ static void unsetRightChild(unsigned char *const bitcodes_root, const size_t &index)
			{bitcodes_root[index] &= ~HAS_RIGHT_CHILD;};

	private:
		GPUTreeNode() {};
		virtual ~GPUTreeNode() {};
		// Explicitly deletes the copy assignment operator and copy constructor
		GPUTreeNode& operator=(GPUTreeNode &source) = delete;	// assignment operator
		GPUTreeNode(GPUTreeNode &node) = delete;	// copy constructor

		// Bitcodes used to indicate presence of left/right children (and potentially other values as necessary) to save space, as bool actually takes up 1 byte, same as a char
		// Without an explicit instantiation, enums don't take up any space
		enum Bitcodes
		{
			HAS_LEFT_CHILD = 0x2,
			HAS_RIGHT_CHILD = 0x1
		};
};

#endif
