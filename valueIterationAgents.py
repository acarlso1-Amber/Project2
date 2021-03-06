# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from multiprocessing.sharedctypes import Value
import mdp, util

from learningAgents import ValueEstimationAgent
import collections
import sys


#Project 2 Question 1
#March 21, 2022
#Travis, Amber, Wen
class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.



        python3.7 autograder.py -q q1
        python3.7 gridworld.py -a value -i 100 -k 10
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()
        self.n = 0

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        allPossibleStates = self.mdp.getStates()

        for i in range(self.iterations):
            aCounterOrSomething = util.Counter()
            for state in allPossibleStates:
                max_val = -1000000
                for action in self.mdp.getPossibleActions(state):
                    bestQValues = self.computeQValueFromValues(state, action)
                    if bestQValues > max_val:
                        max_val = bestQValues
                    aCounterOrSomething[state] = max_val
            self.values = aCounterOrSomething
       

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        Q = 0
        transitions = self.mdp.getTransitionStatesAndProbs(state, action) # list of tuples of --> statePrime, probability

        for statePrime, probability in transitions: 
            value = self.values[statePrime]
            reward = self.mdp.getReward(state, action, statePrime)
            Q += probability * (reward + (value*self.discount))
        return Q


    #Travis, Amber, Wen 3/29/22
    def computeActionFromValues(self, state):
        """
          computes the best action according to the value function given by self.values

          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        if self.mdp.isTerminal(state):
            return None

        actions = self.mdp.getPossibleActions(state)        

        bestAction = actions[0]
        bestQ = -100000

        for action in actions:
            Q = self.computeQValueFromValues(state,action)
            if Q > bestQ:
                bestQ = Q
                bestAction = action
        return bestAction


    def computeBestQ(self,state):
        actions = self.mdp.getPossibleActions(state)        

        qs = []
        for action in actions:
            Q = self.computeQValueFromValues(state,action)
            qs.append(Q)

        if len(qs) == 0:
            return 0

        return max(qs)


    def getPolicy(self, state):
        #print("getPolicy is called")
        return self.computeActionFromValues(state)

    def getAction(self, state):
        #print("getAction is called")
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        #print("getQValue is called")
        return self.computeQValueFromValues(state, action)

#Travis, Wen 3/29/2022
#Question 4
class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
        
    #Travis, Wen 4/5/2022
    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        states = self.mdp.getStates()
        lenStates = len(states)



        for i in range(self.iterations):
            n = i % lenStates
            if not(self.mdp.isTerminal(states[n])):
                action = self.getAction(states[n])
                self.values[states[n]] = self.computeQValueFromValues(states[n], action)

#Travis, Wen 4/5/2022
#Question 5
class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.

          python3.7 gridworld.py -a priosweepvalue -i 1000
          python3.7 autograder.py -q q5
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)


    #Travis, Wen, 4/5/2022
    def runValueIteration(self):
                
        states = self.mdp.getStates()
        predecessorAssociations = {}
        for state in states:
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    transitions = self.mdp.getTransitionStatesAndProbs(state, action)
                    for statePrime,probability in transitions:
                        if (probability != 0.0):
                            if statePrime in predecessorAssociations:
                                predecessorAssociations[statePrime].add(state)
                            else:
                                predecessorAssociations[statePrime] = {state}
        #print(predecessorAssociations)
        
        pq = util.PriorityQueue()
        #Build pq
        for state in states:
            if (not self.mdp.isTerminal(state)):
                v = self.getValue(state)
                q = self.computeBestQ(state)
                diff = abs(v-q)
                pq.update(state,-diff)
        
        #Evaluate
        for i in range(self.iterations):
            if (not pq.isEmpty()):
                state = pq.pop()
                if not self.mdp.isTerminal(state):
                    action = self.getAction(state) 
                    self.values[state] = self.computeQValueFromValues(state, action)
                    predecessors = predecessorAssociations[state]
                    for predecessor in predecessors:
                        v = self.values[predecessor]
                        q = self.computeBestQ(predecessor)
                        diff = abs(v-q)
                        if diff > self.theta:
                            pq.update(predecessor,-diff)