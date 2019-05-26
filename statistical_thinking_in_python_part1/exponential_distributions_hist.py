## Bernoulli trials 
def perform_bernoulli_trials(n, p):
    n_succcess = 0
    for i in range(n):
        random = np.random.random()

        if random > p:
            n_success += 1

    return n_success

# more efficient implementation using numpy

n_defaults = np.random.binomial(100, 0.05, size=10000)

# == 100 events in one sample with 0.05 probability of success, repeated 10000 times


## Exponential distribution for waiting times between events that are Poisson distributed

def successive_poisson(tau1, tau2, size=1):
    t1 = np.random.exponential(tau1, size=size)
    t2 = np.random.exponential(tau2, size=size)
    return t1 + t2


# Draw samples of waiting times: waiting_times
waiting_times = successive_poisson(764, 715, size=100000)

# Make the histogram
plt.hist(waiting_times, bins=100, normed=True, histtype='step')


# Label axes

plt.xlabel("Mean waiting time")
plt.ylabel("PDF")

# Show the plot
plt.show()
