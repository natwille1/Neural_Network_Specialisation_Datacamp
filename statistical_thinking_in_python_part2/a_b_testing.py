# The Civil Rights Act of 1964 was one of the most important pieces of legislation ever passed in the USA.
# Excluding "present" and "abstain" votes, 153 House Democrats and 136 Republicans voted yea.
# However, 91 Democrats and 35 Republicans voted nay. Did party affiliation make a difference in the vote?

# To answer this question, you will evaluate the hypothesis that the party of a House member has no bearing on his or her vote.
# You will use the fraction of Democrats voting in favor as your test statistic and
# evaluate the probability of observing a fraction of Democrats voting in favor at least as small as the observed fraction of 153/244.
# (That's right, at least as small as. In 1964, it was the Democrats who were less progressive on civil rights issues.)
# To do this, permute the party labels of the House voters and then arbitrarily divide them into "Democrats" and "Republicans" and compute the fraction of Democrats voting yea.

# Construct arrays of data: dems, reps
dems = np.array([True] * 153 + [False] * 91)
reps = np.array([True] * 136 + [False] * 35)

def frac_yea_dems(dems, reps):
    """Compute fraction of Democrat yea votes."""
    frac = np.sum(dems[True]) / len(dems)
    return frac

# Acquire permutation samples: perm_replicates
perm_replicates = draw_perm_reps(dems, reps, frac_yea_dems, size=10000)

# Compute and print p-value: p
p = np.sum(perm_replicates <= 153/244) / len(perm_replicates)
print('p-value =', p)


## Perform an A/B test to test if rules changes in baseball resulted in a statistically significant different in
## mean no-hitter time. 
# Compute the observed difference in mean inter-no-hitter times: nht_diff_obs
nht_diff_obs = diff_of_means(nht_dead, nht_live)

# Acquire 10,000 permutation replicates of difference in mean no-hitter time: perm_replicates
perm_replicates = draw_perm_reps(nht_dead, nht_live, diff_of_means, size=10000)


# Compute and print the p-value: p
p = np.sum(perm_replicates <= nht_diff_obs) / len(perm_replicates)
print('p-val =', p)
