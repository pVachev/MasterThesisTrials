from data.getData import fetch_data
from data.getData import FILEPATH
from src.load import *
import os
import tabulate
from src.transform import clean_data
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from hmmlearn import hmm


def main():

    tickers = ["BND", "SPY", "^IRX"]
#    
# 
# clean_data("BND")
#    new_path = os.path.dirname(os.path.realpath(__file__)) + "\\raw\\"
    df = clean_data(tickers)


    seed = 7 

    df = diff_data(df)

    # print(df.head())

    
    cols = ["ExcessLogBND", "ExcessLogSPY"]

    # # plot_returns(bnd,gspc)
    df_m = df
    df_m = df_m.resample("ME").sum()[cols].dropna()
    df_m = df_m[cols]

    print(df_m["ExcessLogBND"].mean())
    print(df_m["ExcessLogSPY"].mean())
    print(df)
    print(df_m)

    X = df_m.to_numpy()

    

    n_states = 2


    model = hmm.GaussianHMM(n_components=n_states, 
                            n_iter=3000, 
                            covariance_type="full",
                            random_state=seed)
    


    model.fit(X)

    states = model.predict(X)
    probs = model.predict_proba(X)
    df_m["state"] = states


    


    g = df_m.groupby(by=["state"])[cols]
    out = pd.concat(
        [
            (100* g.mean()).add_prefix("mean_%_"),
            (100 * g.std(ddof=1)).add_prefix("std_%_"),
            df_m.groupby(["state"]).size().rename("n_obs"),
    ],
    axis=1)

    out.sort_values(f"std_%_{cols[0]}", ascending=False)

    print(out)

    # dist_plot(bnd)

    # qq_normal(bnd)

    # print(bnd)


    # plt.show()
    




#    print(new_path)
    
    # if os.path.exists(FILEPATH):

        # print((__file__))
        # print((FILEPATH))
    
if __name__ == "__main__":

    main()