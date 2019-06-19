# ECDF of improvement from low to high lanes
# Now that you have a metric for improvement going from low- to high-numbered lanes,
# plot an ECDF of this metric. I have put together the swim times of all swimmers who swam a 50 m semifinal
# in a high numbered lane and the final in a low numbered lane, and vice versa.
# The swim times are stored in the Numpy arrays swimtime_high_lanes
# and swimtime_low_lanes. Entry i in the respective arrays are for the same swimmer in the same event.

# Compute the fractional improvement of being in high lane: f
f = (swimtime_low_lanes - swimtime_high_lanes) / swimtime_low_lanes

# Make x and y values for ECDF: x, y
x,y = dcst.ecdf(f)

# Plot the ECDFs as dots
plt.plot(x,y,marker='.', linestyle='none')

# Label the axes and show the plot
plt.xlabel("swimtimes")
plt.ylabel("ECDF")
plt.show()

# Estimation of mean improvement
# You will now estimate how big this current effect is. Compute the mean fractional
# improvement for being in a high-numbered lane versus a low-numbered lane,
# along with a 95% confidence interval of the mean.

# Compute the mean difference: f_mean
f_mean = np.mean(f)

# Draw 10,000 bootstrap replicates: bs_reps
bs_reps = dcst.draw_bs_reps(f, np.mean, size=10000)

# Compute 95% confidence interval: conf_int
conf_int = np.percentile(bs_reps, [2.5, 97.5])

# Print the result
print("""
mean frac. diff.: {0:.5f}
95% conf int of mean frac. diff.: [{1:.5f}, {2:.5f}]""".format(f_mean, *conf_int))

# Hypothesis test: Does lane assignment affect performance?
# Perform a bootstrap hypothesis test of the null hypothesis that the mean fractional
# improvement going from low-numbered lanes to high-numbered lanes is zero. Take the fractional improvement
# as your test statistic,
# and "at least as extreme as" to mean that the test statistic under the null hypothesis is
# greater than or equal to what was observed.

# Shift f: f_shift
f_shift = f - np.mean(f)

# Draw 100,000 bootstrap replicates of the mean: bs_reps
bs_reps = dcst.draw_bs_reps(f_shift, np.mean, size=100000)

# Compute and report the p-value
p_val = np.sum(bs_reps >= f_mean) / 100000
print('p =', p_val)

# Did the 2015 event have this problem?
# You would like to know if this is a typical problem with pools in competitive swimming.
# To address this question, perform a similar analysis for the results of the 2015 FINA World Championships.
# That is, compute the mean fractional improvement for going from lanes 1-3 to lanes 6-8 for
# the 2015 competition, along with a 95% confidence interval on the mean. Also test the hypothesis
# that the mean fractional improvement is zero.
# The arrays swimtime_low_lanes_15 and swimtime_high_lanes_15 have the pertinent data.

# Compute f and its mean
f = (swimtime_low_lanes_15 - swimtime_high_lanes_15) / swimtime_low_lanes_15
f_mean = np.mean(f)

# Draw 10,000 bootstrap replicates
bs_reps = dcst.draw_bs_reps(f, np.mean, size=10000)

# Compute 95% confidence interval
conf_int = np.percentile(bs_reps, [2.5, 97.5])

# Shift f
f_shift = f - f_mean

# Draw 100,000 bootstrap replicates of the mean
bs_reps = dcst.draw_bs_reps(f_shift, np.mean, size=10000)

# Compute the p-value
p_val = np.sum(bs_reps >= f_mean) / 100000

# Print the results
print("""
mean frac. diff.: {0:.5f}
95% conf int of mean frac. diff.: [{1:.5f}, {2:.5f}]
p-value: {3:.5f}""".format(f_mean, *conf_int, p_val))

# EDA: mean differences between odd and even splits
# To investigate the differences between odd and even splits, you first need to define a difference metric.
# In previous exercises, you investigated the improvement of moving from a low-numbered lane to a high-numbered lane,
# defining f = (ta - tb) / ta. There, the ta in the denominator served as our reference time for improvement.
# Here, you are considering both improvement and decline in performance depending on the direction of swimming,
# so you want the reference to be an average. So, we will define the fractional difference as f = 2(ta - tb) / (ta + tb).
#
# Your task here is to plot the mean fractional difference between odd and even splits versus lane number.
# I have already calculated the mean fractional differences for the 2013 and 2015 Worlds for you, and they
# are stored in f_13 and f_15. The corresponding lane numbers are in the array lanes.


# Plot the the fractional difference for 2013 and 2015
plt.plot(lanes, f_13, marker='.', markersize=12, linestyle='none')
plt.plot(lanes,f_15, marker='.', markersize=12, linestyle='none')

# Add a legend
_ = plt.legend((2013, 2015))

# Label axes and show plot
plt.xlabel("lane")
plt.ylabel("frac. diff. (odd-even)")
plt.show()
