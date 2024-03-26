#ifndef PST_TESTER_H
#define PST_TESTER_H

#include <random>
#include <type_traits>

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
		 	template<typename, template<typename, typename, size_t> class, typename, size_t> StaticPSTTemplate,
			typename IDType, size_t num_ID_fields, typename RandNumEng=std::mt19937 mt_eng,
			template<typename> typename FloatDistrib=uniform_real_distribution>
	requires std::is_floating_point<T>::value
struct PSTTester
{
	RandNumEng rand_num_eng;
	FloatDistrib<T> distr;

	TesterClass(T min_val, T max_val)
		: rand_num_eng(0),
		distr(min_val, max_val)
	{};

	TesterClass(size_t rand_seed, T min_val, T max_val)
		: rand_num_eng(rand_seed),
		distr(min_val, max_val)
	{};
};

template <typename T, template<typename, typename, size_t> class PointStructTemplate,
		 	template<typename, template<typename, typename, size_t> class, typename, size_t> StaticPSTTemplate,
			typename IDType, size_t num_ID_fields, typename RandNumEng=std::mt19937 mt_eng,
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
};

#endif
