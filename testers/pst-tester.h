#ifndef PST_TESTER_H
#define PST_TESTER_H

#include <random>
#include <type_traits>

template <typename T, typename RandNumEng=std::mt19937 mt_eng,
			template<typename> typename FloatDistrib=uniform_real_distribution>
	requires std::is_floating_point<T>::value
struct PSTTester
{
	RandNumEng rand_num_eng;
	FloatDistrib<T> distr;
	
	enum TestCodes
	{
		CONSTRUCT,
		REPORT_ALL,
		LEFT_SEARCH,
		RIGHT_SEARCH,
		THREE_SEARCH
	};

	TesterClass(T min_val, T max_val)
		: rand_num_eng(0),
		distr(min_val, max_val)
	{};

	TesterClass(size_t rand_seed, T min_val, T max_val)
		: rand_num_eng(rand_seed),
		distr(min_val, max_val)
	{};
	template <template<typename, typename, size_t> class PointStructTemplate,
			template<typename, template<typename, typename, size_t> class, typename, size_t> StaticPSTTemplate,
			typename IDType, size_t num_IDs>
	void test(size_t num_elems, TestCodes test_type)
	{
		PointStructTemplate<T, IDType, num_IDs> *pt_arr = new PointStructTemplate<T, IDType, num_IDs>[num_elems];

		for (size_t i = 0; i < num_elems; i++)
		{
			pt_arr[i].dim1_val = distr(mt_eng);
			pt_arr[i].dim2_val = distr(mt_eng);
			// Lazy instantiation of value of type IDType from type T
			if constexpr (num_IDs == 1)
				pt_arr[i].id = *const_cast<IDType *>(&distr(mt_eng));
		}


	#ifdef DEBUG
		printArray(std::cout, pt_arr, 0, num_elems);
		std::cout << '\n';
	#endif


		StaticPSTTemplate<T, PointStructTemplate, IDType, num_IDs> *tree = new StaticPSTTemplate<T, PointStructTemplate, IDType, num_IDs>(pt_arr, num_elems);

		if (tree != nullptr)
			std::cout << *tree << '\n';



		delete tree;
		delete[] pt_arr;
	};
};

template <typename T, typename RandNumEng=std::mt19937 mt_eng,
			template<typename> typename IntDistrib=uniform_int_distribution>
	requires std::is_integral<T>::value
struct PSTTester
{
	RandNumEng rand_num_eng;
	IntDistrib<T> distr;

	TesterClass(T min_val, T max_val)
		: rand_num_eng(0),
		distr(min_val, max_val)
	{};

	TesterClass(size_t rand_seed, T min_val, T max_val)
		: rand_num_eng(rand_seed),
		distr(min_val, max_val)
	{};

	template <template<typename, typename, size_t> class PointStructTemplate,
		 	template<typename, template<typename, typename, size_t> class, typename, size_t> StaticPSTTemplate,
			typename IDType, size_t num_IDs>
	void test(size_t num_elems, )
	{
		PointStructTemplate<T, IDType, num_IDs> *pt_arr = new PointStructTemplate<T, IDType, num_IDs>[num_elems];

		for (size_t i = 0; i < num_elems; i++)
		{
			pt_arr[i].dim1_val = distr(mt_eng);
			pt_arr[i].dim2_val = distr(mt_eng);
			// Lazy instantiation of value of type IDType from type T
			if constexpr (num_IDs == 1)
				pt_arr[i].id = *const_cast<IDType *>(&distr(mt_eng));
		}


	#ifdef DEBUG
		printArray(std::cout, pt_arr, 0, num_elems);
		std::cout << '\n';
	#endif


		StaticPSTTemplate<T, PointStructTemplate, IDType, num_IDs> *tree = new StaticPSTTemplate<T, PointStructTemplate, IDType, num_IDs>(pt_arr, num_elems);

		if (tree != nullptr)
			std::cout << *tree << '\n';




		delete tree;
		delete[] pt_arr;
	};
};

#endif
