// As GPU threads cannot access locations smaller than a byte without access conflicts and associated race conditions, make each bitcode take a byte (as CUDA supports global memory access of words of size 1, 2, 4, 8 and 16 B)
template <typename T>
class StaticPSTCPUIter<T>::TreeNode
{
	public:
		// Current index is this-root
		// Addition and subtraction (+, -) have higher precedence than bitshift operators (<<, >>)
		static size_t getLeftChild(const size_t &index) {return (index << 1) + 1;};
		static size_t getRightChild(const size_t &index) {return (index << 1) + 2;};
		static size_t getParent(const size_t &index) {return index - 1 >> 1;};
		static size_t getNumBitcodesPerByte() {return codes_per_byte;};
		// From the specification of C, pointers are const if the const qualifier appears to the right of the corresponding *
		static bool hasChildren(unsigned char * const bitcodes_root, const size_t &index)
		{
			// Bitshift has higher precedence than bitwise operations
			return static_cast<bool> (bitcodes_root[getBitcodeIndex(index)]
										& (HAS_LEFT_CHILD | HAS_RIGHT_CHILD)
											<< getBitcodeOffset(index));
		};
		static bool hasLeftChild(unsigned char * const bitcodes_root, const size_t &index)
		{
			return static_cast<bool> (bitcodes_root[getBitcodeIndex(index)]
										& HAS_LEFT_CHILD << getBitcodeOffset(index));
		};
		static bool hasRightChild(unsigned char * const bitcodes_root, const size_t &index)
		{
			return static_cast<bool> (bitcodes_root[getBitcodeIndex(index)]
										& HAS_RIGHT_CHILD << getBitcodeOffset(index));
		};
		static void setLeftChild(unsigned char * const bitcodes_root, const size_t &index)
			{bitcodes_root[getBitcodeIndex(index)] |= HAS_LEFT_CHILD << getBitcodeOffset(index);};
		static void setRightChild(unsigned char * const bitcodes_root, const size_t &index)
			{bitcodes_root[getBitcodeIndex(index)] |= HAS_RIGHT_CHILD << getBitcodeOffset(index);};
		static void unsetLeftChild(unsigned char * const bitcodes_root, const size_t &index)
			{bitcodes_root[getBitcodeIndex(index)] &= ~HAS_LEFT_CHILD << getBitcodeOffset(index);};
		static void unsetRightChild(unsigned char * const bitcodes_root, const size_t &index)
			{bitcodes_root[getBitcodeIndex(index)] &= ~HAS_RIGHT_CHILD << getBitcodeOffset(index);};

	private:
		TreeNode() {};
		virtual ~TreeNode() {};
		TreeNode& operator=(TreeNode &source) {};	// assignment operator
		TreeNode(TreeNode &node) {};	// copy constructor

		// Helper functions
		static size_t getBitcodeIndex(size_t index)
			{return index / codes_per_byte;};
		// Internal offset of lowest place value in byte of code corresponding to a given node; little-endian ordering is used within each byte
		static size_t getBitcodeOffset(size_t index)
			// Multiplication, division and remainder are all of the same precedence and operate left-to-right
			{return index % codes_per_byte * bits_per_code;};

		// Bitcodes used to indicate presence of left/right children (and potentially other values as necessary) to save space, as bool actually takes up 1 byte, same as a char
		// Without an explicit instantiation, Bitcodes won't take up any space
		enum Bitcodes
		{
			HAS_LEFT_CHILD = 0x2,
			HAS_RIGHT_CHILD = 0x1
		};

		const static size_t bits_per_code = 2;
		const static size_t bits_per_byte = 8;
		const static size_t codes_per_byte = bits_per_byte/bits_per_code;
};
