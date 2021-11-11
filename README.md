# Online Elephant 

Adaption of statistical analysis methods from *Elephant*  to perform online 
analysis within a corporative simulation during run-time.

## Planned Methods
#### Correlations
* Cross-correlation between neuron: -> is it possible to calculate for 100x100 
neurons the approximately 10000 cross-correlation pairs 
within one simulation time step? -> parallelization needed/sufficient?

#### Rate Estimation
* Mean Firing Rate of Neurons
* Instantaneous Rate of Neurons
#### Spike Interval Statistics
* Inter-Spike-Intervall (ISI) of a spiketrain
* (local) coefficient of variation
#### Statistics across spike trains 
* Fanofactor (probably unsuitable for online analysis, because it's a cross trial meassure)
* Complexity / Probability Density (within trial should be possible & across trial probably unsuitable)