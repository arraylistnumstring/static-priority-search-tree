template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
class StaticPSTCPURecur<T, PointStructTemplate, IDType, num_IDs>::TreeNode
{
	public:
		TreeNode();
		TreeNode(PointStructTemplate<T, IDType, num_IDs> &source_data, T median_dim1_val);
		virtual ~TreeNode() {};
		TreeNode& operator=(TreeNode &source);	// assignment operator
		TreeNode(TreeNode &node);	// copy constructor

		// Printing function for printing operator << to use, as private data members must be accessed in the process
		// const keyword after method name indicates that the method does not modify any data members of the associated class
		virtual void print(std::ostream &os) const
		{
			os << '(' << pt.dim1_val << ", " << pt.dim2_val  << "; " << median_dim1_val;
			if constexpr (num_IDs == 1)
				os << "; " << pt.id;
			os << ')';
		};

		void setTreeNode(PointStructTemplate<T, IDType, num_IDs> &source_data, T median_dim1_val);

		// Current index is this-root
		// Addition and subtraction (+, -) have higher precedence than bitshift operators (<<, >>)
		// root needs to be passed into get*() functions as nested class are not connected to any instances of the outer class in C++
		inline TreeNode &getLeftChild(TreeNode *root) const {return root[(this-root << 1) + 1];};
		inline TreeNode &getRightChild(TreeNode *root) const {return root[(this-root << 1) + 2];};
		inline TreeNode &getParent(TreeNode *root) const {return root[this-root-1 >> 1];};
		inline bool hasChildren() const {return static_cast<bool> (code & (HAS_LEFT_CHILD | HAS_RIGHT_CHILD));};
		inline bool hasLeftChild() const {return static_cast<bool> (code & HAS_LEFT_CHILD);};
		inline bool hasRightChild() const {return static_cast<bool> (code & HAS_RIGHT_CHILD);};
		inline void setLeftChild() {code |= HAS_LEFT_CHILD;};
		inline void setRightChild() {code |= HAS_RIGHT_CHILD;};
		inline void unsetLeftChild() {code &= ~HAS_LEFT_CHILD;};
		inline void unsetRightChild() {code &= ~HAS_RIGHT_CHILD;};

		PointStructTemplate<T, IDType, num_IDs> pt;
		T median_dim1_val;
		// bool (1 byte) auto-converts any values other than 0 or 1 to 1, so another type is necessary; as the struct takes the alignment requirement of the largest enclosed data type, and typical use cases mean that T will be at least as big as an int or a float (4B), it is fine to use an unsigned int (4B) to store the bitcode
		unsigned code;

		// Bitcodes used to indicate presence of left/right children (and potentially other values as necessary) to save space, as bool actually takes up 1 byte, same as a char
		// Without an explicit instantiation, Bitcodes won't take up any space
		enum Bitcodes
		{
			HAS_LEFT_CHILD = 0x2,
			HAS_RIGHT_CHILD = 0x1
		};
};

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
StaticPSTCPURecur<T, PointStructTemplate, IDType, num_IDs>::TreeNode::TreeNode()
	// Value initialisation is more efficient with member initialiser lists, as they are not default-initialised before being overriden
	// When no members are explicitly initialised, default-initialisation of non-class variables with automatic or dynamic storage duration produces objects with indeterminate values
	: median_dim1_val(0),
	code(0)
{}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
StaticPSTCPURecur<T, PointStructTemplate, IDType, num_IDs>::TreeNode::TreeNode(PointStructTemplate<T, IDType, num_IDs> &source_data, T median_dim1_val)
	// When other subobjects are explicitly initialised, those that are not are implicit initialised in the same way as objects with static storage duration, i.e. with 0 or nullptr (stated in 6.7.8 (19) of the C++ standard)
	: pt(source_data),
	median_dim1_val(median_dim1_val)
{}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
// typename necessary before StaticPSTCPURecur<T, PointStructTemplate, IDType, num_IDs> in order for two-phase compiler to establish that StaticPSTCPURecur<T, PointStructTemplate, IDType, num_IDs>::TreeNode is a type and therefore a valid return type, rather than a member variable of StaticPSTCPURecur<T, PointStructTemplate, IDType, num_IDs>
typename StaticPSTCPURecur<T, PointStructTemplate, IDType, num_IDs>::TreeNode& StaticPSTCPURecur<T, PointStructTemplate, IDType, num_IDs>::TreeNode::operator=(TreeNode &source)
{
	if (this == &source)
		return *this;	// If the two addresses match, it's the same object

	pt = source.pt;
	median_dim1_val = source.median_dim1_val;
	code = source.code;

	return *this;
}

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
			typename IDType, size_t num_IDs>
void StaticPSTCPURecur<T, PointStructTemplate, IDType, num_IDs>::TreeNode::setTreeNode(PointStructTemplate<T, IDType, num_IDs> &source_data, T median_dim1_val)
{
	pt = source_data;	// assignment operator invoked
	this->median_dim1_val = median_dim1_val;
}
