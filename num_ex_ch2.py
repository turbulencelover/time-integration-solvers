from pandas import DataFrame as df
import numpy as np
from dc_experiments import DCs as dc
from serial_examples import Schemes as scheme
import test_examples as ex
from time import time
from functools import wraps


def test(T, y0, M, N, approach, ff, y_t, solver):
    start = time()
    tt, yy = solver(ff, T, y0, N, M, approach)
    finish = time()
    print(f'Finished in {round(finish-start, 2)} second(s)')

    ts = np.linspace(0, T, N+1)
    y_true = y_t(ts)
    return tt, yy.T, y_true, round(finish-start, 2)


def order():
    NotImplemented


if __name__ == '__main__':

    N = 320
    start, stop = 0, 1
    init_cond = 0
    h = (stop-start)/N
    ts = np.arange(start, stop, h)

    dy_dt = ex.func0
    y = ex.y_exact0
    y0 = 1
    yN = y(stop)

    serial = scheme(v_field=dy_dt,
                    start=start,
                    stop=stop,
                    h=h,
                    init_conditions=y0)

    ts0, fe, fe_time = serial.euler()
    _, rk4, rk4_time = serial.rk4()
    _, ab4, ab4_time = serial.ab4()

    true = np.array([y(t) for t in ts])
    fe_err = np.mean(np.abs(true-fe))
    rk4_err = np.mean(np.abs(true -rk4))
    ab4_err = np.mean(np.abs(true -rk4))

    (ts1, idc_fe3), time_idc_fe3 = dc().idc_fe(a=start, b= stop, alpha = y0, N =N, p= 3, f=dy_dt)
    (ts2, idc_fe6), time_idc_fe6 = dc().idc_fe(a=start, b= stop, alpha = y0, N =N, p= 6, f=dy_dt)
    idc_fe3_err = np.mean(np.abs(true-idc_fe3))
    idc_fe6_err = np.mean(np.abs(true-idc_fe6))

    ridc_fe3, time_ridc_fe3 = dc().ridc_fe(a=start, b= stop, alpha = y0, N =N, p= 4, K = N, f=dy_dt)
    ridc_fe4, time_ridc_fe4 = dc().ridc_fe(a=start, b= stop, alpha = y0, N =N, p= 4, K = int(N/2), f=dy_dt)
    ridc_fe5, time_ridc_fe5 = dc().ridc_fe(a=start, b= stop, alpha = y0, N =N, p= 4, K = int(N/4), f=dy_dt)
    ridc_fe3_err = np.mean(np.abs(true-ridc_fe3[1:]))
    ridc_fe4_err = np.mean(np.abs(true-ridc_fe4[1:]))
    ridc_fe5_err = np.mean(np.abs(true-ridc_fe5[1:]))

    ridc_rk6, time_ridc_rk6 = dc().ridc_rk4(a=start, b= stop, alpha = y0, N =N, p= 4, K = N, f=dy_dt)
    ridc_rk7, time_ridc_rk7 = dc().ridc_rk4(a=start, b= stop, alpha = y0, N =N, p= 4, K = int(N/2), f=dy_dt)
    ridc_rk8, time_ridc_rk8 = dc().ridc_rk4(a=start, b= stop, alpha = y0, N =N, p= 4, K = int(N/4), f=dy_dt)
    ridc_rk6_err = np.mean(np.abs(true-ridc_rk6[1:]))
    ridc_rk7_err = np.mean(np.abs(true-ridc_rk7[1:]))
    ridc_rk8_err = np.mean(np.abs(true-ridc_rk8[1:]))

    ridc_ab9, time_ridc_ab9 = dc().ridc_ab2(a=start, b= stop, alpha = y0, N =N, p= 4, K = N, f=dy_dt)
    ridc_ab10, time_ridc_rk10 = dc().ridc_ab2(a=start, b= stop, alpha = y0, N =N, p= 4, K = int(N/2), f=dy_dt)
    ridc_ab11, time_ridc_rk11 = dc().ridc_ab2(a=start, b= stop, alpha = y0, N =N, p= 4, K = int(N/4), f=dy_dt)
    ridc_ab9_err = np.mean(np.abs(true-ridc_ab9[1:]))
    ridc_ab10_err = np.mean(np.abs(true-ridc_ab10[1:]))
    ridc_ab11_err = np.mean(np.abs(true-ridc_ab11[1:]))
    
    ridc_abM, time_ridc_abM = dc().ridc_abM(T=stop, y0=[y0], N=N, M=4, approach=0, f=dy_dt)
    ridc_abM_err = np.mean(np.abs(true-ridc_abM[1:]))

    ind = ['FE', 'RK4', 'AB4', 
            'IDC3-FE', 'IDC6-FE', 
            'RIDC(4,N)-FE', 'RIDC(4,N/2)-FE', 'RIDC(4,N/4)-FE', 
            'RIDC(4,N)-RK4', 'RIDC(4,N/2)-RK4', 'RIDC(4,N/4)-RK4',
            'RIDC(4,N)-AB2', 'RIDC(4,N/2)-AB2', 'RIDC(4,N/4)-AB2', 'RIDC4_N0_STENCIL']
    

    times = [fe_time, rk4_time, ab4_time, 
                time_idc_fe3, time_idc_fe6,
                time_ridc_fe3, time_ridc_fe4, time_ridc_fe5, 
                time_ridc_rk6, time_ridc_rk7, time_ridc_rk8, 
                time_ridc_ab9, time_ridc_rk10, time_ridc_rk11, time_ridc_abM]

    errors = [fe_err, rk4_err, ab4_err,
                idc_fe3_err, idc_fe6_err, 
                ridc_fe3_err, ridc_fe4_err, ridc_fe5_err, 
                ridc_rk6_err, ridc_rk7_err, ridc_rk8_err, 
                ridc_ab9_err, ridc_ab10_err, ridc_ab11_err, ridc_abM_err]

    n = [N]*len(times) 
    k = [np.nan,np.nan,np.nan,
            np.nan,np.nan,
            320, 160, 80, 
            320, 160, 80,
            320, 160, 80, np.nan ]

    data = {'N': n, 'K': k, 'Time': times, 'Average Error': errors}

    # Creates pandas DataFrame. 
    df_ = df(data, index =ind) 
    
    # print the data 
    print(df_)
    print(df_.to_latex())
