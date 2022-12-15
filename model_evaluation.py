import pandas as pd
from scipy import stats
import scikit_posthocs as sp
import numpy as np

"""

The null hypothesis (H0): The algorithms have NOT s.s. between them

The alternative hypothesis: (Ha): At least one algorithms is has s.s. from the rest.

"""

df = pd.read_csv('algo_performance.csv')
statistic = stats.friedmanchisquare(df['C4.5'], df['1-NN'], df['NaiveBayes'], df['Kernel'], df['CN2'])

"""
statistic=39.91275167785245
pvalue=0.00000004512033059024698

In this example, the test statistic is 39.91275 and the corresponding p-value is 0.00000004. 
Since this p-value is less than 0.05, 0.025 and 0.0125 we can reject the null hypothesis that 
the algorithms have s.s. between them.

In other words, we have sufficient evidence to conclude that the algorithms used 
leads to statistically significant differences.


"""

data = np.array([df['C4.5'], df['1-NN'], df['NaiveBayes'], df['Kernel'], df['CN2']])
sp.posthoc_nemenyi_friedman(df.T)

"""
results: 
        0	        1	        2	        3	        4
0	1.000000	0.038984	0.900000	0.001000	0.092627
1	0.038984	1.000000	0.075768	0.061173	0.900000
2	0.900000	0.075768	1.000000	0.001000	0.163237
3	0.001000	0.061173	0.001000	1.000000	0.024071
4	0.092627	0.900000	0.163237	0.024071	1.000000

The Nemeyi post-hoc test returns the p-values for each pairwise comparison of means. 
From the output we can see the following p-values:

P-value of group 0 vs. group 1: 0.038984
P-value of group 0 vs. group 2: 0.900000
P-value of group 0 vs. group 3: 0.001000
P-value of group 0 vs. group 4: 0.092627

P-value of group 1 vs. group 2: 0.075768
P-value of group 1 vs. group 3: 0.061173
P-value of group 1 vs. group 4: 0.900000

P-value of group 2 vs. group 3: 0.001000
P-value of group 2 vs. group 4: 0.163237

P-value of group 3 vs. group 4: 0.024071


At α = .05, the groups that have S.S are group 0, group 1 ,group 2 and group 3 and group 4.

At α = .025, the groups that have S.S are group 0, group 2 and group 3 and group 4.

At α = .0125, the groups that have S.S are group 0, group 2 and group 3.


"""

(90000, 37)
(180000,38)
(270000,44)
(360000,53)
(450000,20)
(540000,22)
(630000,54)
(671000,48)
(761000,47)
(821000,51)
(911000,39)
(1001000,38)
(1091000,43)
(1181000,43)
(1271000,29)
(1361000,38)
(1451000,51)
(1541000,43)
(1630500,52)
(1720500,42)
(1810500,46)
(1900500,24)
(1967000,40)
(2057000,45)
(2117500,53)
(2207500,49)
(2297500,47)
(2387500,35)
(2477500,32)
(2567500,36)
(2624000,58)
(2648500,77)
(2738500,38)
(2775000,54)
(2865000,51)
(2931500,52)
(3019000,47)
(3109000,52)
(3198500,19)
(3221000,56)
(3311000,41)
(3401000,57)
(3491000,50)
(3581000,53)
(3630500,45)
(3720500,33)
(3783500,49)
(3873500,45)
(3963000,49)
(4053000,49)
(4093500,44)
(4155500,54)
(4245500,51)
(4335500,46)
(4425500,23)