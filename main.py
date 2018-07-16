import numpy as np
import argparse
import copy
import statistics
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class Q_Player:
    def __init__(self, max_price, alpha, delta, epsilon, theta):
        # the row is the price of opponent and the colum is the one of
        # itself
        self.Q = np.random.uniform(size = [max_price + 1, max_price + 1])
        self.max_price = max_price
        # Learning rate
        self.alpha = alpha
        # Discount rate
        self.delta = delta
        # decaying rate
        self.theta = theta
        # Indicate this is a Q model
        self.Qmodel = True
        # the prob of exploitation
        self.epsilon = epsilon

    def offer(self, p_oppo_prev):
        if np.random.binomial(1, self.epsilon):
            self.epsilon = self.epsilon * (1-self.theta)
            # Uniform price
            # because the random integer is from [l,h), we need to plus one
            # here
            return np.random.randint(self.max_price + 1)
        else:
            self.epsilon = self.epsilon * (1-self.theta)
            # exploitation
            return np.argmax(self.Q[p_oppo_prev])

    def updateQ(self, market, p_oppo_past, p_self_curr, p_oppo_react):
        self.Q[p_oppo_past][p_self_curr] = \
                (1-self.alpha) * self.Q[p_oppo_past][p_self_curr] + \
                self.alpha * (market.profit(p_self_curr, p_oppo_past) + \
                        self.delta * market.profit(p_self_curr, p_oppo_react) + \
                              (self.delta ** 2) * max(self.Q[p_oppo_react]) )

class Myopic_Player:
    def __init__(self):
        #indicate this is not a Q model
        self.Qmodel = False
    def offer(self, p_oppo_prev):
        return max(p_oppo_prev - 1, 1)

class market:
    def __init__(self, intercept):
        self.intercept = intercept
    def profit(self, p_self, p_oppo):
        if p_self < p_oppo:
            return 1. * (self.intercept - p_self) * p_self
        elif p_self > p_oppo:
            return 0.
        else:
            return 0.5 * (self.intercept-p_self) * p_self

def similate(player1, player2, market, num_of_runs, periods, max_price):
    # First element is for player 1
    results = ([], [])
    # Price for last period and this period
    p_last = np.random.randint(max_price + 1)
    p_curr = np.random.randint(max_price + 1)
    for i in range(num_of_runs):
        results[0].append([])
        results[1].append([])
        # make copy of players
        p1 = copy.deepcopy(player1)
        p2 = copy.deepcopy(player2)
        for j in range(periods - 1):
            # stage of player 1
            # In this half, p_last is offered by player 1, and p_curr is offered
            # by player 2
            offer = p1.offer(p_curr)
            profit = market.profit(offer, p_curr)
            results[0][i].append({'price':offer, 'profit':profit})
            if p2.Qmodel:
                p2.updateQ(market, p_last, p_curr, offer)
            # update price
            p_last = p_curr
            p_curr = offer
            # Stage of player 2
            # Here, p_last is by player 2 and p_curr is by player 1
            offer = p2.offer(p_curr)
            profit = market.profit(offer, p_curr)
            results[1][i].append({'price':offer, 'profit':profit})
            if p1.Qmodel:
                p1.updateQ(market, p_last, p_curr, offer)
            # update price
            p_last = p_curr
            p_curr = offer
    return results

def averagmatrix(results):
    ave = []
    for i in range(len(results[0])):
        ave.append({'price':statistics.mean([results[j][i]['price'] for j in \
                                           range(len(results))]),
                   'profit':statistics.mean([results[j][i]['profit'] for j in \
                                            range(len(results))])}) 
    return ave


def main():
    parser = argparse.ArgumentParser(description='replicate Klein 2018')
    parser.add_argument('--num-of-runs', type = int, default = 10, metavar = 'N',
                        help = 'number of the simulations (default: 100)')
    parser.add_argument('--periods', type = int, default = 1000, metavar = 'N',
                        help = 'how may times the players offer price each (default: 100)')
    parser.add_argument('--max-price', type = int, default = 12, metavar = 'N',
                        help = 'the largest possible price (default: 12)')
    parser.add_argument('--theta', type = float, default = 0.001, metavar =
                        'f', help = 'the decaying rate of exploration \
                        probability (default: 0.001)' )
    parser.add_argument('--delta', type = float, default = 0.9, metavar = 'f',
                        help = 'discouting rate (decault: 0.9)')
    parser.add_argument('--epsilon0', type = float, default = 1.0, metavar =
                        'f', help = 'initial exploration probability \
                        (default :1)')
    parser.add_argument('--alpha', type = float, default = 0.9, metavar = 'f',
                       help = 'learning rate in the Q-learning')
    args = parser.parse_args()

    # define market. Assume the intercept equals the max price
    mkt = market(args.max_price)
    # define players
    p1 = Q_Player(args.max_price, args.alpha, args.delta, args.epsilon0, args.theta)
    p2 = Myopic_Player()

    results = similate(p1, p2, mkt, args.num_of_runs, args.periods, args.max_price)
    
    averesults = averagmatrix(results[0])
    
    plt.plot([l['profit'] for l in averesults])
    plt.show()


if __name__ == '__main__':
    main()
