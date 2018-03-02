#Copyright 2017 Martin Ueding
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import math

def unitprint(value, error, value_digits=3, error_digits=1, allowed_hang=3):
    if value == 0:
        value_log = 0
    else:
        value_log = int(math.floor(math.log(abs(value), 10)))

    if error is None or error == 0:
        digits=value_digits
        if abs(value_log) > allowed_hang:
            value_mantissa = ("{:."+str(digits-1)+"f}").format(value * 10**(- value_log))
            error_mantissa = None
            exponent = value_log
        else:
            value_mantissa = ("{:."+str(max(digits-1 - value_log, 0))+"f}").format(value)
            error_mantissa = None
            exponent = 0
    else:
        error_log = int(math.floor(math.log(abs(error), 10)))

        difference = value_log - error_log

        value_dis = value * 10**(- value_log)
        error_dis = error * 10**(-difference - error_log)
        exp = value_log

        if abs(value_log) > allowed_hang:
            here_digits = error_digits - 1 + max(difference, 0)

            value_mantissa = ("{:."+str(here_digits)+"f}").format(value_dis)
            error_mantissa = ("{:."+str(here_digits)+"f}").format(error_dis)
            exponent = exp
        else:
            here_digits = max(error_digits - 1 -error_log, 0)

            value_mantissa = ("{:."+str(here_digits)+"f}").format(value)
            error_mantissa = ("{:."+str(here_digits)+"f}").format(error)
            exponent = 0

    if error_mantissa is None:
        if exponent == 0:
            return "{}".format(value_mantissa)
        else:
            return "{}e{}".format(value_mantissa, exponent)
    else:
        if exponent == 0:
            return "{} +- {}".format(value_mantissa, error_mantissa)
        else:
            return "{} +- {} e{}".format(value_mantissa, error_mantissa, exponent)
if __name__=="__main__":
    pass
