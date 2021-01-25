from pandas import DataFrame as DF

import dc_experiments as dc
import serial_examples as serial
import test_examples as ex


def test(T, y0, M, N, approach, ff, y_t):
    start = time.perf_counter()
    tt, yy = solver(ff, T, y0, N, M, approach)
    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')

    ts = np.linspace(0, T, N+1)
    y_true = y_t(ts)
    return tt, yy.T, y_true, round(finish-start, 2)



if __name__ = '__main__':
    N = 320

    

    
