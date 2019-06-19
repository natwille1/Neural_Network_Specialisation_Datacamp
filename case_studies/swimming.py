# Graphical EDA of men's 200 free heats
# In the heats, all contestants swim, the very fast and the very slow.
# To explore how the swim times are distributed, plot an ECDF of the men's 200 freestyle.

# Generate x and y values for ECDF: x, y
x, y = dcst.ecdf(mens_200_free_heats)

# Plot the ECDF as dots
plt.plot(x,y, marker = '.', linestyle='none')


# Label axes and show plot
plt.xlabel("time (s)")
plt.ylabel("ECDF")
plt.show()
#
# 200 m free time with confidence interval
# Now, you will practice parameter estimation and computation of confidence intervals by computing the mean and median swim time for the men's 200 freestyle heats.
# The median is useful because it is immune to heavy tails in the distribution of swim times, such as the slow swimmers in the heats.
# mens_200_free_heats is still in your namespace.

# Compute mean and median swim times
mean_time = np.mean(mens_200_free_heats)
median_time = np.median(mens_200_free_heats)

# Draw 10,000 bootstrap replicates of the mean and median
bs_reps_mean = dcst.draw_bs_reps(mens_200_free_heats, np.mean, size=10000)
bs_reps_median = dcst.draw_bs_reps(mens_200_free_heats, np.median, size=10000)


# Compute the 95% confidence intervals
conf_int_mean = np.percentile(bs_reps_mean, [2.5, 97.5])
conf_int_median = np.percentile(bs_reps_median, [2.5, 97.5])

# Print the result to the screen
print("""
mean time: {0:.2f} sec.
95% conf int of mean: [{1:.2f}, {2:.2f}] sec.

median time: {3:.2f} sec.
95% conf int of median: [{4:.2f}, {5:.2f}] sec.
""".format(mean_time, *conf_int_mean, median_time, *conf_int_median))

# EDA: finals versus semifinals
# First, you will get an understanding of how athletes' performance changes from the semifinals to the finals by computing the fractional improvement from the semifinals to finals and plotting an ECDF of all of these values.
#
# The arrays final_times and semi_times contain the swim times of the respective rounds.
# The arrays are aligned such that final_times[i] and semi_times[i] are for the same swimmer/event.
# If you are interested in the strokes/events, you can check out the data frame df in your namespace, which has more detailed information, but is not used in the analysis.
#

# Compute fractional difference in time between finals and semis
f = (semi_times - final_times) / semi_times

# Generate x and y values for the ECDF: x, y
x, y = dcst.ecdf(f)

# Make a plot of the ECDF
plt.plot(x,y, marker='.', linestyle='none')

# Label axes and show plot
_ = plt.xlabel('f')
_ = plt.ylabel('ECDF')
plt.show()

# Parameter estimates of difference between finals and semifinals
# Compute the mean fractional improvement from the semifinals to finals, along with a 95% confidence interval of the mean.
# The Numpy array f that you computed in the last exercise is in your namespace.

# Mean fractional time difference: f_mean
f_mean = np.mean(f)

# Get bootstrap reps of mean: bs_reps
bs_reps = dcst.draw_bs_reps(f, np.mean, size=10000)

# Compute confidence intervals: conf_int
conf_int = np.percentile(bs_reps, [2.5, 97.5])

# Report
print("""
mean frac. diff.: {0:.5f}
95% conf int of mean frac. diff.: [{1:.5f}, {2:.5f}]""".format(f_mean, *conf_int))

# Generating permutation samples
# As you worked out in the last exercise, we need to generate a permutation sample by randomly swapping corresponding entries in the semi_times and final_times array.
# Write a function with signature swap_random(a, b) that returns arrays where random indices have the entries in a and b swapped.

def swap_random(a, b):
    """Randomly swap entries in two arrays."""
    # Indices to swap
    swap_inds = np.random.random(size=len(a)) < 0.5

    # Make copies of arrays a and b for output
    a_out = np.copy(a)
    b_out = np.copy(b)

    # Swap values
    a_out[swap_inds] = b[swap_inds]
    b_out[swap_inds] = a[swap_inds]

    return a_out, b_out


# Hypothesis test: Do women swim the same way in semis and finals?
# Test the hypothesis that performance in the finals and semifinals are
# identical using the mean of the fractional improvement as your test statistic.
# The test statistic under the null hypothesis is considered to be at least as extreme
# as what was observed if it is greater than or equal to f_mean, which is already in your namespace.
# The semifinal and final times are contained in the numpy arrays semi_times and final_times.

# Set up array of permutation replicates
perm_reps = np.empty(1000)

for i in range(1000):
    # Generate a permutation sample
    semi_perm, final_perm = swap_random(semi_times, final_times)

    # Compute f from the permutation sample
    f = (semi_perm - final_perm) / semi_perm

    # Compute and store permutation replicate
    perm_reps[i] = np.mean(f)

# Compute and print p-value
print('p =', np.sum(perm_reps >= f_mean) / 1000)

# EDA: Plot all your data
# To get a graphical overview of a data set, it is often useful to plot all of your data.
# In this exercise, plot all of the splits for all female swimmers in the 800 meter heats.
# The data are available in a Numpy arrays split_number and splits.
# The arrays are organized such that splits[i,j] is the split time for swimmer i for split_number[j].

# Plot the splits for each swimmer
for splitset in splits:
    _ = plt.plot(split_number, splitset, lw=1, color='lightgray')

# Compute the mean split times
mean_splits = np.mean(splits, axis=0)

# Plot the mean split times
plt.plot(split_number, mean_splits, marker='.', linewidth=3, markersize=12)

# Label axes and show plot
_ = plt.xlabel('split number')
_ = plt.ylabel('split time (s)')
plt.show()

# Perform regression
slowdown, split_3 = np.polyfit(split_number, mean_splits, deg=1)

# Compute pairs bootstrap
bs_reps, _ = dcst.draw_bs_pairs_linreg(split_number, mean_splits, size=10000)

# Compute confidence interval
conf_int = np.percentile(bs_reps, [2.5, 97.5])

# Plot the data with regressions line
_ = plt.plot(split_number, mean_splits, marker='.', linestyle='none')
_ = plt.plot(split_number, slowdown * split_number + split_3, '-')

# Label axes and show plot
_ = plt.xlabel('split number')
_ = plt.ylabel('split time (s)')
plt.show()

# Print the slowdown per split
print("""
mean slowdown: {0:.3f} sec./split
95% conf int of mean slowdown: [{1:.3f}, {2:.3f}] sec./split""".format(
    slowdown, *conf_int))

# Hypothesis test: are they slowing down?
# Now we will test the null hypothesis that the swimmer's split time is not at all correlated with the distance they are at in the swim.
# We will use the Pearson correlation coefficient (computed using dcst.pearson_r()) as the test statistic.

# Observed correlation
rho = dcst.pearson_r(split_number, np.mean(splits, axis=0))

# Initialize permutation reps
perm_reps_rho = np.empty(10000)

# Make permutation reps
for i in range(10000):
    # Scramble the split number array
    scrambled_split_number = np.random.permutation(split_number)

    # Compute the Pearson correlation coefficient
    perm_reps_rho[i] = dcst.pearson_r(scrambled_split_number, np.mean(splits,axis=0))

# Compute and print p-value
p_val = np.sum(perm_reps_rho >= rho) / len(perm_reps_rho)
print('p =', p_val)
