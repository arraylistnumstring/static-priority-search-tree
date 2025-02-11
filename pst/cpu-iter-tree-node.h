#ifndef CPU_ITER_TREE_NODE_H
#define CPU_ITER_TREE_NODE_H

// Identical to GPUTreeNode, with the exception of missing __forceinline__, __host__ and __device__ keywords to allow for use of StaticPSTCPUIter on machines without CUDA or a GPU

class CPUIterTreeNode
{
	public:
		// Current index is this-root
		// Addition and subtraction (+, -) have higher precedence than bitshift operators (<<, >>)
		static size_t getLeftChild(const size_t index) {return (index << 1) + 1;};
		static size_t getRightChild(const size_t index) {return (index << 1) + 2;};
		static size_t getParent(const size_t index) {return index - 1 >> 1;};
		// From the specification of C, pointers are const if the const qualifier appears to the right of the corresponding *
		static bool hasChildren(const unsigned char bitcode)
		{
			// Bitwise and (&) has higher precedence than bitwise or (|)
			return static_cast<bool> (bitcode & (HAS_LEFT_CHILD | HAS_RIGHT_CHILD));
		};
		static bool hasLeftChild(const unsigned char bitcode)
			{return static_cast<bool> (bitcode & HAS_LEFT_CHILD);};
		static bool hasRightChild(const unsigned char bitcode)
			{return static_cast<bool> (bitcode & HAS_RIGHT_CHILD);};
		static void setLeftChild(unsigned char *const bitcodes_root, const size_t index)
			{bitcodes_root[index] |= HAS_LEFT_CHILD;};
		static void setRightChild(unsigned char *const bitcodes_root, const size_t index)
			{bitcodes_root[index] |= HAS_RIGHT_CHILD;};
		static void unsetLeftChild(unsigned char *const bitcodes_root, const size_t index)
			{bitcodes_root[index] &= ~HAS_LEFT_CHILD;};
		static void unsetRightChild(unsigned char *const bitcodes_root, const size_t index)
			{bitcodes_root[index] &= ~HAS_RIGHT_CHILD;};

	private:
		CPUIterTreeNode() {};
		virtual ~CPUIterTreeNode() {};
		// Explicitly deletes the copy assignment and copy constructors
		CPUIterTreeNode& operator=(CPUIterTreeNode &source) = delete;	// assignment operator
		CPUIterTreeNode(CPUIterTreeNode &node) = delete;	// copy constructor

		// Bitcodes used to indicate presence of left/right children (and potentially other values as necessary) to save space, as bool actually takes up 1 byte, same as a char
		// Without an explicit instantiation, Bitcodes won't take up any space
		enum Bitcodes
		{
			HAS_LEFT_CHILD = 0x2,
			HAS_RIGHT_CHILD = 0x1
		};
};

#endif
