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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

#Travis Mewborne
#Project 2 Question 1
#March 21, 2022
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

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        print("runValueIteration called")


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
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        print("cqvf transitions", transitions)
        Q=0
        for transition in transitions:
            #print("cqfv ",transition)
            statePrime = transition[0]
            reward = self.mdp.getReward(state, action, statePrime)
            probability = transition[1]
            #value = self.computeActionFromValues(transition[0])
            #value = self.computeQValueFromValues(statePrime)
            #value = self.getValue(statePrime)
            value = self.values[statePrime]
            #print("cqfv ",reward)
            #print("cqfv ",probability)
            #print("cqfv ",value)
            #if (value == "None"):
            #    Q += probability * reward
            #else:
            Q += probability * (reward + value*self.discount)
        self.values[state] = Q
        return Q
        print("computeQValueFromValues called")
        #return self.values[state]
        util.raiseNotDefined()

    #Travis, Amber, Wen 3/22/22
    def computeActionFromValues(self, state):
        print("computeActionFromValues called")
        """
          computes the best action according to the value function given by self.values

          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        print("---hi----")

        #Returning none if final state
        if self.mdp.isTerminal(state):
            return "None"

        actions = self.mdp.getPossibleActions(state)

        #if (actions[0]=="exit"):
        #    transitions = self.mdp.getTransitionStatesAndProbs(state, actions[0])
        #    return self.mdp.getReward(state, actions[0], transitions[0][0])

        
        print("values:\t",self.values)
        print("state:\t",state)
        print(self.mdp.isTerminal(state))
        
        print("Possible actions from state:\t",actions)
        print("Extracted Q Value: ", self.values[state])
        print("States: ", self.mdp.getStates())
        for action in actions:
            print("\tFOR ACTION:",action)
            transitions = self.mdp.getTransitionStatesAndProbs(state, action)
            #print("\t\tTransition states ", transitions)
            #for transition in transitions:
            #    print("\t\t\t",transition,"\tReward: ",self.mdp.getReward(state, action, transition[0]))
            #print("\tReward ", self.mdp.getReward(state, action, transitions[0][0]))
        print("Actually doing stuff")
        


        results = []
        
        for action in actions:
            Q = self.computeQValueFromValues(state,action)
            results.append((action,Q))

        resultsDec = (sorted(results, key = lambda x: x[1]))
        resultsAsc = resultsDec[::-1]

        print(resultsAsc    )

        QStar = resultsAsc[0][1]
        BestAction = resultsAsc[0][0]


        """
        AStart = results[0][0]
        QStar = results[0][1]
        for result in results:
            a = result[0]
            q = result[1]
            if q>QStar:
                q=QStar
                AStart=a"""
        
       

        print ("results ",results, "BEST",BestAction,"//",QStar)
        print("---bye---")

        return(BestAction)

        

        

        
        
        #return actions[0]
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

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

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

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
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

