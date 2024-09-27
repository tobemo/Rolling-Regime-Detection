# Rolling regime detection
My implementation of "Robust Rolling Regime Detection (R2-Rd): A Data-Driven Perspective of Financial Markets" by Ali Hirsa, Sikun Xu and Satyan Malhotra.

When time series data does not have temporally stable regimes, fitting one hidden Markov model on all data will prove to be suboptimal.
This code base implements a regime classifier which updates over time, adding regimes as needed.
See paper for more details.

## Adjustments
The paper uses distance metrics both for label assignment an regime emergence but I don't agree on the second point, or at least not with how I have implemented the transition cost matrix.

Given the original model_t, the new model_t+1 and the new model with an added regime model_t+1_n+1; and knowing that in fact a regime should be added:
If model_t and model_t+1 emit the same exact regime labels for X_t+1 then the distance between both will be 0. Meanwhile model_t+1_n+1 will, by definition, emit different labels (the original ones plus a new one) so the distance between this one and model_t will also be greater than 0.

In summary, if model_1_a is equally 'bad' as model_0 at detecting a new regime, but it does so in exactly the same manner, then it's cost is low, and model_t+1 is favoured.

![](distance_metric_flaw.png)


Conclusion, as of 10/09/24, is to use cost solely for regime mapping, and aic&bic for model selection.


## Known issues
Numpy's 2. relaese seems to have broken hmmlearn. All predicted regimes default to one value only.
Fixed by downgrading numpy to the latest 1. release: 1.26.4.

https://github.com/hmmlearn/hmmlearn/issues/557

