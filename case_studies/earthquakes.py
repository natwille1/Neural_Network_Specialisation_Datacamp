# Parkfield earthquake magnitudes
# As usual, you will start with EDA and plot the ECDF of the magnitudes of
# earthquakes detected in the Parkfield region from 1950 to 2016.
# The magnitudes of all earthquakes in the region from the ANSS ComCat are stored in the Numpy array mags.
#
# When you do it this time, though, take a shortcut in generating the ECDF.
# You may recall that putting an asterisk before an argument in a function
# splits what follows into separate arguments. Since dcst.ecdf() returns two values,
# we can pass them as the x, y positional arguments to plt.plot() as plt.plot(*dcst.ecdf(data_you_want_to_plot)).
#
# You will use this shortcut in this exercise and going forward.

# Make the plot
plt.plot(*dcst.ecdf(mags), marker='.', linestyle='none')

# Label axes and show plot
plt.xlabel('magnitude')
plt.ylabel('ECDF')
plt.show()

# Computing the b-value
# The b-value is a common metric for the seismicity of a region.
# You can imagine you would like to calculate it often when working
# with earthquake data. For tasks like this that you will do often,
# it is best to write a function! So, write a function with signature
# b_value(mags, mt, perc=[2.5, 97.5], n_reps=None) that returns the
# b-value and (optionally, if n_reps is not None) its confidence interval
# for a set of magnitudes, mags. The completeness threshold is given by mt.
# The perc keyword argument gives the percentiles for the lower and upper bounds
# of the confidence interval, and n_reps is the number of bootstrap replicates to use in computing the confidence interval.

def b_value(mags, mt, perc=[2.5, 97.5], n_reps=None):
    """Compute the b-value and optionally its confidence interval."""
    # Extract magnitudes above completeness threshold: m
    m = mags[mags >= mt]

    # Compute b-value: b
    #b = (m-mt)*np.log(10)
    b = (np.mean(m)-mt)*np.log(10)

    # Draw bootstrap replicates
    if n_reps is None:
        return b
    else:
        m_bs_reps = dcst.draw_bs_reps(m,np.mean, size=n_reps)

        # Compute b-value from replicates: b_bs_reps
        b_bs_reps = (m_bs_reps - mt) * np.log(10)

        # Compute confidence interval: conf_int
        conf_int = np.percentile(b_bs_reps, perc)

        return b, conf_int

# The b-value for Parkfield
# The ECDF is effective at exposing roll-off, as you could see
# below magnitude 1. Because there are plenty of earthquakes above magnitude 3,
# you can use mt = 3 as your completeness threshold. With this completeness
# threshold, compute the b-value for the Parkfield region from 1950 to 2016,
# along with the 95% confidence interval. Print the results to the screen.
# The variable mags with all the magnitudes is in your namespace.
# Overlay the theoretical Exponential CDF to verify that the Parkfield region follows the Gutenberg-Richter Law.

# Compute b-value and confidence interval
b, conf_int = b_value(mags, mt, perc=[2.5, 97.5], n_reps=10000)

# Generate samples to for theoretical ECDF
m_theor = np.random.exponential(b/np.log(10), size=100000) + mt

# Plot the theoretical CDF
_ = plt.plot(*dcst.ecdf(m_theor))

# Plot the ECDF (slicing mags >= mt)
_ = plt.plot(*dcst.ecdf(mags[mags >= mt]), marker='.', linestyle='none')

# Pretty up and show the plot
_ = plt.xlabel('magnitude')
_ = plt.ylabel('ECDF')
_ = plt.xlim(2.8, 6.2)
plt.show()

# Report the results
print("""
b-value: {0:.2f}
95% conf int: [{1:.2f}, {2:.2f}]""".format(b, *conf_int))


# Interearthquake time estimates for Parkfield
# In this exercise, you will first compute the best estimates for the parameters
# for the Exponential and Gaussian models for interearthquake times. You will then plot the theoretical CDFs
# for the respective models along with the formal ECDF of the actual Parkfield interearthquake times.


# Compute the mean time gap: mean_time_gap
mean_time_gap = np.mean(time_gap)

# Standard deviation of the time gap: std_time_gap
std_time_gap = np.std(time_gap)

# Generate theoretical Exponential distribution of timings: time_gap_exp
time_gap_exp = np.random.exponential(mean_time_gap, size=10000)

# Generate theoretical Normal distribution of timings: time_gap_norm
time_gap_norm = np.random.normal(mean_time_gap, std_time_gap, size=10000)

# Plot theoretical CDFs
_ = plt.plot(*dcst.ecdf(time_gap_exp))
_ = plt.plot(*dcst.ecdf(time_gap_norm))

# Plot Parkfield ECDF
_ = plt.plot(*dcst.ecdf(time_gap, formal=True, min_x=-10, max_x=50))

# Add legend
_ = plt.legend(('Exp.', 'Norm.'), loc='upper left')

# Label axes, set limits and show plot
_ = plt.xlabel('time gap (years)')
_ = plt.ylabel('ECDF')
_ = plt.xlim(-10, 50)
plt.show()


# When will the next big Parkfield quake be?
# The last big earthquake in the Parkfield region was on the evening of September 27, 2004 local time.
# \Your task is to get an estimate as to when the next Parkfield quake will be, assuming
# the Exponential model and also the Gaussian model. In both cases, the best estimate
# is given by the mean time gap, which you computed in the last exercise to be 24.62 years,
# meaning that the next earthquake would be in 2029. Compute 95% confidence intervals on when
# the next earthquake will be assuming an Exponential distribution parametrized by mean_time_gap
# you computed in the last exercise. Do the same assuming a Normal distribution parametrized by
# mean_time_gap and std_time_gap.


# Draw samples from the Exponential distribution: exp_samples
exp_samples = np.random.exponential(mean_time_gap, size=100000)

# Draw samples from the Normal distribution: norm_samples
norm_samples = np.random.normal(mean_time_gap, std_time_gap, size=100000)

# No earthquake as of today, so only keep samples that are long enough
exp_samples = exp_samples[exp_samples > today - last_quake]
norm_samples = norm_samples[norm_samples > today - last_quake]

# Compute the confidence intervals with medians
conf_int_exp = np.percentile(exp_samples, [2.5, 50, 97.5]) + last_quake
conf_int_norm = np.percentile(norm_samples, [2.5, 50, 97.5]) + last_quake

# Print the results
print('Exponential:', conf_int_exp)
print('     Normal:', conf_int_norm)


# Computing the K-S statistic
# Write a function to compute the Kolmogorov-Smirnov statistic
# from two datasets, data1 and data2, in which data2 consists of samples
# from the theoretical distribution you are comparing your data to.
# Note that this means we are using hacker stats to compute the K-S statistic
# for a dataset and a theoretical distribution, not the K-S statistic for two empirical datasets.
# Conveniently, the function you just selected for computing values of the formal ECDF is given as dcst.ecdf_formal().

def ks_stat(data1, data2):
    # Compute ECDF from data: x, y
    x,y = dcst.ecdf(data1)

    # Compute corresponding values of the target CDF
    cdf = dcst.ecdf_formal(x, data2)

    # Compute distances between concave corners and CDF
    D_top = y - cdf

    # Compute distance between convex corners and CDF
    D_bottom = cdf - y + 1/len(data1)

    return np.max((D_top, D_bottom))


def draw_ks_reps(n, f, args=(), size=10000, n_reps=10000):
    # Generate samples from target distribution
    x_f = f(*args, size=size)

    # Initialize K-S replicates
    reps = np.empty(n_reps)

    # Draw replicates
    for i in range(n_reps):
        # Draw samples for comparison
        x_samp = f(*args, size=n)

        # Compute K-S statistic
        reps[i] = dcst.ks_stat(x_samp, x_f)

    return reps

# The K-S test for Exponentiality
# Test the null hypothesis that the interearthquake times
# of the Parkfield sequence are Exponentially distributed.
# That is, earthquakes happen at random with no memory of when the
# last one was. Note: This calculation is computationally
# intensive (you will draw more than 108 random numbers), so it will take about 10 seconds to complete.

# Draw target distribution: x_f
x_f = np.random.exponential(mean_time_gap, size=10000)

# Compute K-S stat: d
d = dcst.ks_stat(x_f, time_gap)

# Draw K-S replicates: reps
reps = dcst.draw_ks_reps(len(time_gap), np.random.exponential,
                         args=(mean_time_gap,), size=10000, n_reps=10000)

# Compute and print p-value
p_val = np.sum(reps >= d) / 10000
print('p =', p_val)


# EDA: Plotting earthquakes over time
# Make a plot where the y-axis is the magnitude and
# the x-axis is the time of all earthquakes in Oklahoma
# between 1980 and the first half of 2017. Each dot in the plot represents a single earthquake.
# The time of the earthquakes, as decimal years, is stored in the Numpy array time,
# and the magnitudes in the Numpy array mags.

# Plot time vs. magnitude
plt.plot(time, mags, marker='.', linestyle='none', alpha=0.1)

# Label axes and show the plot
plt.xlabel("time (year)")
plt.ylabel("magnitude")
plt.show()

# Estimates of the mean interearthquake times
# The graphical EDA in the last exercise shows an obvious change
# in earthquake frequency around 2010. To compare, compute the mean time
# between earthquakes of magnitude 3 and larger from 1980 through 2009 and
# also from 2010 through mid-2017. Also include 95% confidence
# intervals of the mean. The variables dt_pre and dt_post respectively
# contain the time gap between all earthquakes of magnitude at least 3
# from pre-2010 and post-2010 in units of days.

# Compute mean interearthquake time
mean_dt_pre = np.mean(dt_pre)
mean_dt_post = np.mean(dt_post)

# Draw 10,000 bootstrap replicates of the mean
bs_reps_pre = dcst.draw_bs_reps(dt_pre, np.mean, size=10000)
bs_reps_post = dcst.draw_bs_reps(dt_post, np.mean, size=10000)

# Compute the confidence interval
conf_int_pre = np.percentile(bs_reps_pre, [2.5, 97.5])
conf_int_post = np.percentile(bs_reps_post, [2.5, 97.5])

# Print the results
print("""1980 through 2009
mean time gap: {0:.2f} days
95% conf int: [{1:.2f}, {2:.2f}] days""".format(mean_dt_pre, *conf_int_pre))

print("""
2010 through mid-2017
mean time gap: {0:.2f} days
95% conf int: [{1:.2f}, {2:.2f}] days""".format(mean_dt_post, *conf_int_post))

# Hypothesis test: did earthquake frequency change?
# Obviously, there was a massive increase in earthquake frequency
# once wastewater injection began. Nonetheless, you will still
# do a hypothesis test for practice. You will not test the hypothesis
# that the interearthquake times have the same distribution before
# and after 2010, since wastewater injection may affect the distribution.
# Instead, you will assume that they have the same mean. So, compute
# the p-value associated with the hypothesis that the pre- and post-2010
#  interearthquake times have the same mean, using the mean of pre-2010 time
#  gaps minus the mean of post-2010 time gaps as your test statistic.

# Compute the observed test statistic
mean_dt_diff = mean_dt_pre - mean_dt_post

# Shift the post-2010 data to have the same mean as the pre-2010 data
dt_post_shift = dt_post - mean_dt_post + mean_dt_pre

# Compute 10,000 bootstrap replicates from arrays
bs_reps_pre = dcst.draw_bs_reps(dt_pre, np.mean, size=10000)
bs_reps_post = dcst.draw_bs_reps(dt_post_shift, np.mean, size=10000)

# Get replicates of difference of means
bs_reps = bs_reps_pre - bs_reps_post

# Compute and print the p-value
p_val = np.sum(bs_reps >= mean_dt_diff) / 10000
print('p =', p_val)

# EDA: Comparing magnitudes before and after 2010
# Make an ECDF of earthquake magnitudes from 1980 through 2009.
# On the same plot, show an ECDF of magnitudes of earthquakes from 2010 through mid-2017.
# The time of the earthquakes, as decimal years, are stored in the Numpy array time and the magnitudes in the Numpy array mags.

# Get magnitudes before and after 2010
mags_pre = mags[time < 2010]
mags_post = mags[time >= 2010]

# Generate ECDFs
plt.plot(*dcst.ecdf(mags_pre), marker='.', linestyle='none')
plt.plot(*dcst.ecdf(mags_post), marker='.', linestyle='none')

# Label axes and show plot
_ = plt.xlabel('magnitude')
_ = plt.ylabel('ECDF')
plt.legend(('1980 though 2009', '2010 through mid-2017'), loc='upper left')
plt.show()

# Quantification of the b-values
# Based on the plot you generated in the previous exercise, you
# can safely use a completeness threshold of mt = 3. Using this threshold,
# compute b-values for the period between 1980 and 2009 and for 2010 through mid-2017.
# The function b_value() you wrote last chapter, which computes the b-value and
# confidence interval from a set of magnitudes and completeness threshold,
# is available in your namespace, as are the numpy arrays mags_pre and mags_post from the last exercise, and mt.

# Compute b-value and confidence interval for pre-2010
b_pre, conf_int_pre = b_value(mags_pre, mt, perc=[2.5, 97.5], n_reps=10000)

# Compute b-value and confidence interval for post-2010
b_post, conf_int_post =  b_value(mags_post, mt, perc=[2.5, 97.5], n_reps=10000)

# Report the results
print("""
1980 through 2009
b-value: {0:.2f}
95% conf int: [{1:.2f}, {2:.2f}]

2010 through mid-2017
b-value: {3:.2f}
95% conf int: [{4:.2f}, {5:.2f}]
""".format(b_pre, *conf_int_pre, b_post, *conf_int_post))

# Hypothesis test: are the b-values different?
# Perform the hypothesis test sketched out on the previous exercise.

# Only magnitudes above completeness threshold
mags_pre = mags_pre[mags_pre >= mt]
mags_post = mags_post[mags_post >= mt]

# Observed difference in mean magnitudes: diff_obs
diff_obs = np.mean(mags_post) - np.mean(mags_pre)

# Generate permutation replicates: perm_reps
perm_reps = dcst.draw_perm_reps(mags_post, mags_pre, dcst.draw_perm_reps, size=10000)

# Compute and print p-value
p_val = np.sum(perm_reps < diff_obs) / 10000
print('p =', p_val)

# Only magnitudes above completeness threshold
mags_pre = mags_pre[mags_pre >= mt]
mags_post = mags_post[mags_post >= mt]

# Observed difference in mean magnitudes: diff_obs
diff_obs = np.mean(mags_post) - np.mean(mags_pre)

# Generate permutation replicates: perm_reps
perm_reps = dcst.draw_perm_reps(mags_post, mags_pre, dcst.diff_of_means, size=10000)

# Compute and print p-value
p_val = np.sum(perm_reps < diff_obs) / 10000
print('p =', p_val)
