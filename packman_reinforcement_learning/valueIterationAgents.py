# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
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
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0

    self.old_val = self.values.copy()
    states = mdp.getStates()
    for i in range(iterations):
      for state in states:
        actions = mdp.getPossibleActions(state)
        if not mdp.isTerminal(state):
          actionVal = -float('inf')
          for action in actions:
            qvalue = self. computeQValueFromValues(state,action)
            actionValue = max(actionValue,qValue)
          self.values[state] = actionValue
      self.oldValues = self.values.copy()
     
    "*** YOUR CODE HERE ***"
    
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
    qValue = 0
    for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
      qValue += prob * (self.discount * self.oldValues[nextState] + self.mdp.getReward(state, action, nextState))
    return qValue

  def computeActionFromValues(self, state):
      """
        The policy is the best action in the given state
        according to the values currently stored in self.values.

        You may break ties any way you see fit.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return None.
      """
      "*** YOUR CODE HERE ***"
      actions, maxValue, decision = self.mdp.getPossibleActions(state), -float('inf'), None
      for action in actions:
          actionValue = self.computeQValueFromValues(state, action)
          if actionValue > maxValue:
              maxValue = actionValue
              decision = action
      return decision


  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    "*** YOUR CODE HERE ***"
    return self.computeQValueFromValues(state, action)

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    "*** YOUR CODE HERE ***"
    return self.computeActionFromValues(state)

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)


