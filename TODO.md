- [x] test base
- [ ] test vhmm
    - [ ] implementation
        - [x] init
        - [x] config
        - [x] json
    - [ ] functionality
        - [ ] fit & predict "correctly"
        - [x] transition cost
        - [x] mapping
        - [x] mapper
        - [~] mapped prediction / couldn't test with predictions of more than one class

- [x] FIGURE OUT HOW TO CREATE TEST DATA THAT RESULTS IN REPEATABLE OUTCOMES
    - this was in response to weird outcomes using earthquake data
	  which turns out to be a numpy compatability issue
    - just downgraded np

- [ ] test regime
    - [x] properties
        - [x] with no models
        - [x] with models
    - [x] getitem
    - [x] initial fit
        - [x] specified regime
        - [x] [specified regimes]
        - [x] -1
    - [ ] fit
	- [x] extend transmat
	- [x] extend startprob
	- [x] transition cost matrix
		- [x] n
		- [x] n+1
	- [x] transition cost total
	- [x] comparing cost
        - [x] keeping number of regimes
        - [x] adding regime
    - [x] computing transition threshold
        - [x] too little values
	    - [x] less than 8 values
        - [x] enough values
        - [x] setting and getting
    - [x] restoring
        - [x] with no models
        - [x] with models
	    - [x] n_components as a list
            - [x] with no previous models
            - [x] with previous models
	- [x] self.config works
    - **function tests**
        - [ ] right regime detection
        - [ ] right tracking of regimes
        - [ ] predict
            - [ ] mapper is set
            - [ ] mapper is correct
            - [ ] mapping is constant over many models
        - [ ] new regime
        - [ ] regime disappearing?
- [ ] handle cost matrix being infeasible when optimizing
- [ ] ! train multiple models and use cheapest one