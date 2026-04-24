import logging
import math

import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s
        self._state_decision_map = {}

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))

        if counts_sum == 0:
            valids = self.game.getValidMoves(canonicalBoard, 1)
            valid_sum = np.sum(valids)
            if valid_sum == 0:
                probs = np.ones(self.game.getActionSize(), dtype=np.float32) / self.game.getActionSize()
            else:
                probs = valids / valid_sum
        else:
            probs = [x / counts_sum for x in counts]

        return probs

    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        s_ptr = canonicalBoard
        path = []
        depth = 0
        v=0

        while depth < 50:
            depth += 1
            s = self.game.stringRepresentation(s_ptr)

            if s not in self.Es:
                self.Es[s] = self.game.getGameEnded(s_ptr, 1)
            if self.Es[s] != 0:
                v = -self.Es[s]
                break

            if s not in self.Ps:
                self.Ps[s], v = self.nnet.predict(s_ptr)
                valids = self.game.getValidMoves(s_ptr, 1)
                self.Ps[s] = self.Ps[s] * valids
                sum_Ps = np.sum(self.Ps[s])

                if sum_Ps > 0:
                    self.Ps[s] /= sum_Ps
                else:
                    valid_sum = np.sum(valids)
                    if valid_sum > 0:
                        self.Ps[s] = valids.astype(np.float32) / valid_sum
                    else:
                        self.Ps[s] = np.zeros_like(self.Ps[s], dtype=np.float32)
                        self.Vs[s] = valids
                        self.Ns[s] = 0
                        v = 0
                        break

                self.Vs[s] = valids
                self.Ns[s] = 0
                v = -v
                break

            valids = self.Vs[s]
            cur_best = -float('inf')
            best_act = -1

            for a in range(self.game.getActionSize()):
                if valids[a]:
                    if (s, a) in self.Qsa:
                        u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                                    1 + self.Nsa[(s, a)])
                    else:
                        u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)

                    if u > cur_best:
                        cur_best = u
                        best_act = a

            a = best_act
            if a == -1:
                v = 0
                break
            path.append((s, a))

            next_s, next_player = self.game.getNextState(s_ptr, 1, a)
            s_ptr = self.game.getCanonicalForm(next_s, next_player)

        if depth >= 1000:
            v = 0

        for s, a in reversed(path):
            v = -v
            if (s, a) in self.Qsa:
                self.Qsa[(s, a)] = (
                                           self.Nsa[(s, a)] * self.Qsa[(s, a)] + v
                                   ) / (self.Nsa[(s, a)] + 1)
                self.Nsa[(s, a)] += 1
            else:
                self.Qsa[(s, a)] = v
                self.Nsa[(s, a)] = 1

            if s in self.Ns:
                self.Ns[s] += 1
            else:
                self.Ns[s] = 1

        return -v