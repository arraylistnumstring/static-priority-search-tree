#ifndef ERR_CHK_H
#define ERR_CHK_H

#include <stdexcept>
#include <string>

void throwErr(std::string err_str)
{
	throw std::runtime_error(err_str);
};

#endif
