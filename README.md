### Figure 1. Analysis code for behavior log (performance rate, TTR) and plotting the lick raster plot



author: Eunji Jung 



#### log file structure

1st column: Subject mouse, Date, weight/

3rd-4th column: Event, port code

 	- Sound event

 		211: R1-5kHz (Go)

 		212: R1-10kHz (No-go)

 		2122: R2-10kHz (Go)

 		2112: R2-5kHz (No-go)

 	- Delay (555)

 	- Response (1: lick)

 	- ITI\_1 (666): 2sec

 	- ITI\_2 (777): 2sec

 	- Hit (R1: 351, R2: 352)

 	- Miss (R1: 451, R2: 452)

 	- FA (R1: 311, R2: 312)

 	- CR (R1: 411, R2: 412)

 	- BlockChangeCode (888): between rules and CS/Training stages



##### 

##### Parameter setting

delay : 1000 (ms)

CS\_type:  Indicates whether the session includes a classical conditioning stage between contingency changes

 	- session with CS: CS\_type == 1

 	- session w/o CS: CS\_type == 0



##### Stim-related variables

Stim\_Tra: stimulus onset time, R1-Go, R1-Nogo, R2-Go, and R2-Nogo (in order)

Stim\_Tra\_correct: Correct choice (Hit, CR) == 1, incorrect choice (Miss, FA) == 0

Stim\_Tra\_lick:  Indicates whether a lick occurred during the response window (1 = lick, 0 = no lick)

Stim\_tra\_perform: Performance outcomes in the following order

 	- R1-Hit, R1-Miss, R1-CR, R1-FA, R2-Hit, R2-Miss, R2-CR, R2-FA

 	- 1st column: stim onset time

 	- 2nd column: individual lick time -pre to response window relative to stimulus onset.

\#Note: The same structure applies to variables prefixed with \[Stim\_CS\_\*\*\*]





##### Behavior performance

Stim\_All\_perform\_num: trial number of Hit, Miss, CR, FA in R1 to R2

 	- 1st row: after TTR

 	- 2nd row: before TTR

 	= 3rd row: total (sum of 1st and 2nd rows)

performcri, consec\_tnum:  Used to calculate TTR (performance rate over 0.75 for three consecutive trials)

