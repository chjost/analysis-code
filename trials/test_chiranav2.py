#!/usr/bin/python
import chiral_analysis_v2 as chirana
import numpy as np

def main():
    test_data_addition = chirana.ChirAna("chiral_analysis_test",["mk","m_pi","mu_l","mu_s","beta"])
    datadict= {"mk":np.full(10,0.25),"m_pi":np.full(10,0.12)}
    metadict = {"mu_l":0.001,"mu_s":0.185,"beta":1.9}
    test_data_addition.add_data(datadict,metadict)
    print(test_data_addition.data)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
