# CVST
Traffic flow prediction with CVST data

## TODO

### Data cleaning part

- [x] Process all three months data in matrix format
- [x] Pick the locations with three months data
- [x] Combine all the three months data into one file
- [x] Choose the data with the biggest traffic flow as prediction set

### Algorithms part

ARIMA

- [x] Visualize the data
- [x] Stationarize the data
- [ ] ACF & PACF error analysis(ACF of the series gives correlations between x_t and x_{t-h})
    - [x] Plot
    - [x] season = # of samples per hour * 24
    - [x] ARIMA(p, d, q) × (P, D, Q)S, p = non-seasonal AR order, d = non-seasonal differencing,
     q = non-seasonal MA order, P = seasonal AR order, D = seasonal differencing, Q = seasonal MA order, 
     and S = time span of repeating seasonal pattern.
- [x] Build the ARIMA model and do predicition
- [x] Tuning the parameters and calculate the accuracy

- [ ] Do the same for prediction in each day

SVR-RBF

- [ ] 