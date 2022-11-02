import numpy as np

arms = np.random.random(size=20)
priors = [(10, 10, 1) for i in range(20)]

epsilon = 0.5

total_rewards = 0

for i in range(100):
    if np.random.random() > epsilon:
        probas = [(n*a / (a+b)) for a, b, n in priors]
        ordering = np.argsort(probas)
        selection = ordering[-1]

        reward = np.random.binomial(arms[selection], 1)
    else:

        selection = np.random.randint(0, 20)
        reward = np.random.binomial(1, arms[selection])

    if reward > 0:

        a, b, n = priors[selection]
        priors[selection] = (a+1, b, n)
        total_rewards += 1

    else:
        a, b, n = priors[selection]
        priors[selection] = (a, b+1, n)
